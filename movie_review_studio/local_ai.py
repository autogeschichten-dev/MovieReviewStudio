"""Sequential loopback vision and optional CPU transcription for the director."""
from base64 import b64encode
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
import sys
from time import monotonic
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from .analysis import atomic_json
from .media import identity
from .recap_upstream import frames, transcribe
from .recap_upstream._common import run_cmd

OLLAMA = "http://127.0.0.1:11434"
DEFAULT_MODEL = "moondream:latest"
FIELDS = ("characters", "actions", "environment", "objects", "expressions", "visible_facts", "uncertain")
PROMPT = (
    "Describe this image only. Return a JSON object with these keys and short arrays of strings: "
    + ", ".join(FIELDS)
    + ". Put only directly visible features in the factual fields. Put guesses in uncertain. "
      "Do not infer identities, dialogue, culture, intent, plot or events outside this still. "
      "Use empty arrays for unknown fields."
)


def request(route, payload=None, timeout=5):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    from .director import NoRedirect
    opener = build_opener(ProxyHandler({}), NoRedirect)
    try:
        with opener.open(Request(OLLAMA + route, data=body, headers={"Content-Type": "application/json"}),
                         timeout=timeout) as response:
            return json.loads(response.read(8_000_000))
    except HTTPError as error:
        raise RuntimeError(f"Local Ollama returned HTTP {error.code} for {route}.") from None
    except (URLError, ValueError, TimeoutError):
        raise RuntimeError("Could not read the local Ollama service at 127.0.0.1:11434.") from None


def discover():
    version = request("/api/version").get("version", "unknown")
    models = request("/api/tags").get("models", [])
    return version, {item["name"]: item.get("digest", "") for item in models if item.get("name")}


def describe_image(path, model, source_time, *, keep_alive="2m"):
    image = Path(path).read_bytes()
    if not image.startswith((b"\xff\xd8", b"\x89PNG\r\n\x1a\n")):
        raise ValueError("Vision input must be an actual JPEG or PNG image.")
    output = request("/api/generate", {
        "model": model, "prompt": PROMPT, "images": [b64encode(image).decode("ascii")],
        "format": "json", "stream": False, "keep_alive": keep_alive,
        "options": {"num_ctx": 2048, "num_predict": 240, "temperature": 0},
    }, timeout=90)
    raw = output.get("response")
    try:
        content = json.loads(raw)
    except (TypeError, ValueError):
        raise RuntimeError("Moondream did not return valid structured observations.") from None
    if not isinstance(content, dict):
        raise RuntimeError("Moondream did not return an observation object.")
    values = {}
    for key in FIELDS:
        value = content.get(key, [])
        if isinstance(value, str):
            value = [value]
        values[key] = [item.strip()[:200] for item in value[:6] if isinstance(item, str) and item.strip()] if isinstance(value, list) else []
    if not any(values[field] for field in FIELDS if field != "uncertain"):
        raise RuntimeError("Moondream returned no visible observations.")
    return {"timestamp": source_time, "image_bytes": len(image), "image_sha256": sha256(image).hexdigest(), **values}


def release_model(model):
    active = request("/api/ps").get("models", [])
    if any(row.get("name", row.get("model")) == model for row in active):
        request("/api/generate", {"model": model, "keep_alive": 0, "stream": False}, timeout=10)


class VisionBatch:
    def __init__(self, movie, analysis, cache, model, digest):
        self.movie, self.analysis, self.cache = movie, analysis, cache
        self.model, self.digest = model, digest
        self.source_key = identity(movie.path)
        self.loaded = False

    def describe(self, clips, progress):
        result = {}
        started, processed = monotonic(), 0
        for index, clip in enumerate(clips, 1):
            stamp = f"{self.source_key}:{self.model}:{self.digest}:{clip.start}:{clip.end}:v1"
            saved = self.cache / f"ollama-{sha256(stamp.encode()).hexdigest()}.json"
            progress(f"Visual AI analysis · {index}/{len(clips)}")
            if not saved.exists():
                moment = float((clip.start + clip.frames // 2) / self.movie.fps)
                with tempfile.TemporaryDirectory(prefix="mrs-vision-", dir=self.cache) as folder:
                    frame = Path(folder) / "frame.jpg"
                    frames.extract_frames(self.movie.path, [max(0, moment + self.analysis.offset)],
                                          frame, width=384, labels=False)
                    self.loaded = True
                    description = describe_image(frame, self.model, moment)
                description["source_in"] = float(clip.start / self.movie.fps)
                description["source_out"] = float(clip.end / self.movie.fps)
                atomic_json(saved, description)
                processed += 1
                eta = (monotonic() - started) / processed * (len(clips) - index)
                progress(f"Visual AI analysis · {index}/{len(clips)} · about {math.ceil(eta / 60)} min remaining")
            result[clip.start] = json.loads(saved.read_text(encoding="utf-8"))
        return result

    def unload(self):
        if self.loaded:
            request("/api/generate", {"model": self.model, "keep_alive": 0, "stream": False}, timeout=10)
            self.loaded = False
        else:
            release_model(self.model)


def whisper_python():
    return os.getenv("MRS_WHISPER_PYTHON", "").strip()


def dialogue(movie, cache, progress, model="base"):
    if model not in {"base", "small"}:
        raise ValueError("Choose base or small for faster-whisper.")
    path = movie.path
    sidecar = next((path.with_suffix(ext) for ext in (".srt", ".vtt") if path.with_suffix(ext).is_file()), None)
    stamp = ":".join([identity(path), identity(sidecar) if sidecar else "embedded", model,
                      whisper_python(), os.getenv("MRS_WHISPER_CACHE", "")])
    saved = cache / f"dialogue-31-{sha256(stamp.encode()).hexdigest()}.json"
    progress("Dialogue transcription · checking cached subtitles or transcript")
    if saved.exists():
        return json.loads(saved.read_text(encoding="utf-8"))
    for source in (("srt", "embedded") if sidecar else ("embedded",)):
        output = transcribe.transcribe_media(path, srt_path=sidecar, provider=source)
        if output["status"] == "ok":
            atomic_json(saved, output)
            return output
    python = whisper_python()
    if not python or not movie.has_audio:
        return {"provider": "none", "language": None, "segments": []}
    progress(f"Dialogue transcription · faster-whisper {model} on CPU")
    worker = (Path(sys.executable).parent / "movie_review_studio" / "whisper_worker.py"
              if getattr(sys, "frozen", False) else Path(__file__).with_name("whisper_worker.py"))
    checkpoints = saved.with_suffix("")
    checkpoints.mkdir(exist_ok=True)
    completed = run_cmd([python, str(worker), str(path), model, str(checkpoints)], check=False)
    if completed.returncode:
        raise RuntimeError("faster-whisper failed. Check MRS_WHISPER_PYTHON and the local model setup.")
    output = json.loads(completed.stdout)
    atomic_json(saved, output)
    return output
