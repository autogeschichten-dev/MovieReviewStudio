"""Local, read-only Ollama benchmark. Writes a report even if a provider is missing."""
import ctypes
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from threading import Event, Thread
from time import monotonic

from . import local_ai
from .media import ffmpeg, inspect
from .recap_upstream import frames


def ram_used():
    if sys.platform != "win32":
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in
            ("total", "available", "page", "page_available", "virtual", "virtual_available", "extended")
        ]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return (status.total - status.available) / 1048576
    return None


def gpu_used():
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        result = subprocess.run([smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=3, creationflags=flags)
        return [int(line.strip()) for line in result.stdout.splitlines()] if not result.returncode else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def peak(samples):
    valid = [value for value in samples if value is not None]
    return max(valid) if valid else "unavailable"


def diagnostic(destination, video=None):
    destination = Path(destination)
    started = monotonic()
    log = ["Movie Review Studio V3.1 local AI diagnostic",
           "Measurements are made on this computer. No movie or frame is sent outside loopback."]
    stop = Event()
    ram, gpu = [ram_used()], [gpu_used()]
    def collect():
        while not stop.wait(.8):
            ram.append(ram_used())
            gpu.append(gpu_used())
    sampler = Thread(target=collect, daemon=True)
    sampler.start()
    ok = False
    model = os.getenv("MRS_DIAG_VISION_MODEL", local_ai.DEFAULT_MODEL)
    try:
        version, models = local_ai.discover()
        log += [f"Ollama version: {version}", f"Installed models: {', '.join(models)}"]
        if model not in models:
            raise RuntimeError(f"{model} is not installed.")
        log.append(f"Vision model: {model} ({models[model][:24]})")
        with tempfile.TemporaryDirectory(prefix="mrs-ai-diagnostic-") as directory:
            root = Path(directory)
            fixture = root / "fixture.mp4"
            ffmpeg("-f", "lavfi", "-i", "color=blue:s=384x256:r=4:d=4",
                   "-vf", "drawbox=x=30:y=60:w=105:h=105:color=red:t=fill",
                   "-c:v", "libx264", str(fixture))
            sources = [(fixture, [1.0, 2.0], True)]
            movie = None
            if video:
                movie = inspect(Path(video))
                count = max(1, min(640, int(os.getenv("MRS_DIAG_SAMPLES", "12"))))
                sources.append((movie.path, [float(movie.duration) * (i + .5) / count for i in range(count)], False))
            try:
                for source, times, synthetic in sources:
                    durations = []
                    log.append(f"Source: {source.name} · {len(times)} sampled stills")
                    for i, stamp in enumerate(times):
                        file = root / f"sample-{i}.jpg"
                        frames.extract_frames(source, [stamp], file, width=384, labels=False)
                        begin = monotonic()
                        row = local_ai.describe_image(file, model, stamp)
                        seconds = monotonic() - begin
                        durations.append(seconds)
                        log.append(f"{stamp:.3f}s source · {seconds:.2f}s request · {row['image_bytes']} bytes · SHA-256 {row['image_sha256']}")
                        facts = " ".join(text for key in local_ai.FIELDS if key != "uncertain" for text in row[key])
                        if synthetic:
                            hits = sum(bool(re.search(rf"\b{color}\b", facts, re.I)) for color in ("red", "blue"))
                            log.append(f"Color check {hits}/2 (red rectangle, blue background). Review the description for false additions.")
                        log.append(json.dumps(row, ensure_ascii=True))
                    log.append(f"Mean response: {sum(durations)/len(durations):.2f}s per image. First request includes model warm-up.")
                loaded = local_ai.request("/api/ps").get("models", [])
                for active in loaded:
                    if active.get("name") == model:
                        log.append(f"Ollama size_vram: {active.get('size_vram', 'unavailable')} bytes")
            finally:
                local_ai.request("/api/generate", {"model": model, "keep_alive": 0, "stream": False}, timeout=10)
            if movie and movie.has_audio:
                begin = monotonic()
                result = local_ai.dialogue(movie, root, lambda message: log.append(message),
                                           os.getenv("MRS_DIAG_WHISPER_MODEL", "base"))
                elapsed = monotonic() - begin
                log.append(f"Dialogue: {result['provider']} · {result['language']} · {len(result['segments'])} cues · {elapsed:.2f}s")
                if result["provider"] == "faster-whisper":
                    log.append(f"Transcription speed: {movie.duration / max(.001, elapsed):.2f}x source real time.")
            else:
                log.append("Transcription not measured. Supply an authorized video with audio and configure faster-whisper.")
        ok = True
    except Exception as error:
        log.append(f"ERROR: {type(error).__name__}: {error}")
    finally:
        stop.set()
        sampler.join(4)
        ram.append(ram_used())
        gpu.append(gpu_used())
        log += [f"System RAM used, baseline / sampled peak (MiB): {ram[0]} / {peak(ram)}",
                f"NVIDIA memory used, baseline / sampled maximum GPU (MiB): {gpu[0]} / {peak([max(x) for x in gpu if x])}",
                f"Total elapsed: {monotonic() - started:.2f}s", f"API diagnostic: {'completed' if ok else 'failed'}"]
        destination.write_text("\n".join(log) + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(diagnostic(*sys.argv[1:]))
