"""Sequential loopback vision and optional CPU transcription for the director."""
from base64 import b64encode
from hashlib import sha256
import json
import re
import os
from pathlib import Path
import tempfile
import sys
from time import monotonic
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from .analysis import atomic_json
from .media import identity
from .recap_upstream import frames
from .recap_upstream._common import run_cmd

OLLAMA = "http://127.0.0.1:11434"
DEFAULT_MODEL = "moondream:latest"
FIELDS = ("characters", "actions", "environment", "objects", "expressions", "visible_facts", "uncertain")
PROMPT = "Describe what you see in this image in one or two sentences. Mention visible people, actions, objects and surroundings."
RETRY_PROMPT = "What is visible in this picture? Give a short description."
ALIASES = ("description", "caption", "scene", "summary", "observation")


class ObservationError(RuntimeError):
    pass


def request(route, payload=None, timeout=5, trace=None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    from .director import NoRedirect
    opener = build_opener(ProxyHandler({}), NoRedirect)
    if trace:
        safe = dict(payload or {})
        if "images" in safe:
            safe["images"] = [{"base64_characters": len(image)} for image in safe["images"]]
        trace({"route": route, "request": safe})
    try:
        with opener.open(Request(OLLAMA + route, data=body, headers={"Content-Type": "application/json"}),
                         timeout=timeout) as response:
            raw = response.read(8_000_000).decode("utf-8", errors="replace")
            if trace:
                trace({"http_status": response.status, "body": raw})
            output = json.loads(raw)
            if not isinstance(output, dict):
                raise ValueError("Unexpected response type")
            return output
    except HTTPError as error:
        if trace:
            trace({"http_status": error.code, "body": error.read(65536).decode("utf-8", errors="replace")})
        raise RuntimeError(f"Local Ollama returned HTTP {error.code} for {route}.") from None
    except (URLError, TimeoutError, OSError) as error:
        reason = getattr(error, "reason", error)
        if trace:
            trace({"transport_error": str(reason)})
        if isinstance(reason, TimeoutError):
            raise RuntimeError(f"Local Ollama timed out on {route}.") from None
        raise RuntimeError("Could not reach Ollama at 127.0.0.1:11434.") from None
    except ValueError:
        raise RuntimeError(f"Ollama returned an invalid HTTP response for {route}.") from None


def discover():
    version = request("/api/version").get("version", "unknown")
    models = request("/api/tags").get("models", [])
    return version, {item["name"]: item.get("digest", "") for item in models if item.get("name")}


def decode_image(image):
    import cv2
    import numpy as np
    if not image.startswith((b"\xff\xd8", b"\x89PNG\r\n\x1a\n")):
        raise ValueError("Vision input must be a JPEG or PNG image.")
    pixels = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if pixels is None or min(pixels.shape[:2]) < 16:
        raise ValueError("Vision input is corrupt or too small.")
    return pixels


def normalize_observation(raw):
    if not isinstance(raw, str) or not raw.strip():
        raise ObservationError("Moondream returned an empty description.")
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.I)
    if fenced:
        text = fenced[1].strip()
    try:
        content = json.loads(text)
    except ValueError:
        if text.startswith(("{", "[", "```")):
            raise ObservationError("Moondream returned incomplete structured text.") from None
        content = text
    values = {key: [] for key in FIELDS}

    def strings(value):
        if isinstance(value, str):
            value = [value]
        return [str(item).strip() for item in value[:6]
                if isinstance(item, str) and item.strip()] if isinstance(value, list) else []

    def acceptable(line):
        return (bool(re.search(r"[^\W\d_]{2,}", line)) and
                not re.fullmatch(r"[\s\d.,():xXyYwWhH=+%\[\]\-]+", line) and
                not re.fullmatch(r"(?:unknown|none|null|n/?a|nothing(?: visible)?|no description)[.! ]*", line, re.I) and
                not re.match(r"(?:sorry|i (?:cannot|can't|can not|don't|am unable)|no image|unable to)", line, re.I))

    if isinstance(content, dict):
        for key in FIELDS:
            values[key] = [line for line in strings(content.get(key)) if acceptable(line)]
        free = [line for alias in ALIASES for line in strings(content.get(alias))]
    else:
        free = strings(content)
    description = []
    for line in free:
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            if not acceptable(sentence):
                continue
            if re.search(r"\b(?:maybe|perhaps|probably|possibly|seems|appears to|might be|could be)\b", sentence, re.I):
                values["uncertain"].append(sentence)
            else:
                description.append(sentence)
    values["description"] = " ".join(description)
    if not values["description"] and not any(values[key] for key in FIELDS if key != "uncertain"):
        raise ObservationError("Moondream returned no usable visual description.")
    return values


def describe_image(path, model, source_time, *, keep_alive="2m", prompt=PROMPT, trace=None):
    image = Path(path).read_bytes()
    decode_image(image)
    digest = sha256(image).hexdigest()
    if trace:
        trace({"jpeg_or_png_bytes": len(image), "image_sha256": digest, "source_time": source_time})
    payload = {
        "model": model, "prompt": prompt, "images": [b64encode(image).decode("ascii")],
        "stream": False, "keep_alive": keep_alive,
        "options": {"num_ctx": 2048, "num_predict": 256, "temperature": 0},
    }
    kwargs = {"trace": trace} if trace else {}
    output = request("/api/generate", payload, timeout=60, **kwargs)
    response = output.get("response")
    if output.get("done_reason") == "length" and isinstance(response, str):
        if response.lstrip().startswith(("{", "[", "```")):
            raise ObservationError("Moondream hit its output limit before completing the structured observation.")
        complete = list(re.finditer(r"[.!?。！？](?=\s|$)", response))
        if not complete:
            raise ObservationError("Moondream hit its output limit before completing a description.")
        response = response[:complete[-1].end()]
    values = normalize_observation(response)
    return {"timestamp": source_time, "image_bytes": len(image), "image_sha256": digest, **values}


def usable_frame(path, movie, source_time):
    import cv2
    from .analysis import measure
    pixels = decode_image(path.read_bytes())
    scaled = cv2.resize(pixels, (320, 180), interpolation=cv2.INTER_AREA)
    _, _, black, unusable, text = measure(scaled, None)
    edge = source_time < min(90, movie.duration * .06) or source_time > movie.duration * .85
    return not (black or unusable or (edge and text))


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

    def describe(self, clips, progress, observe=None, *, cache_only=False, allow_empty=False):
        result = {}
        failure = ""
        provider_failures = 0
        self.failed = 0
        self.total = len(clips)
        started = monotonic()
        for index, clip in enumerate(clips, 1):
            if clip.production_card:
                self.failed += 1
                continue
            stamp = f"{self.source_key}:{self.model}:{self.digest}:{clip.start}:{clip.end}:v1"
            saved = self.cache / f"ollama-{sha256(stamp.encode()).hexdigest()}.json"
            preview = saved.with_suffix(".jpg")
            source_time = float((clip.start + clip.frames // 2) / self.movie.fps)
            progress(f"Visual AI analysis · {index}/{self.total} · source {source_time:.2f}s")
            if saved.exists():
                description = json.loads(saved.read_text(encoding="utf-8"))
            elif cache_only:
                self.failed += 1
                continue
            else:
                description = None
                attempted_provider = False
                middle = clip.start + clip.frames // 2
                alternate = clip.start + clip.frames // 4
                with tempfile.TemporaryDirectory(prefix="mrs-vision-", dir=self.cache) as folder:
                    frame = Path(folder) / "frame.jpg"
                    for number, frame_index in enumerate((middle, alternate)):
                        source_time = float(frame_index / self.movie.fps)
                        progress(f"Visual AI analysis · {index}/{self.total} · source {source_time:.2f}s")
                        frame.unlink(missing_ok=True)
                        try:
                            frames.extract_frames(self.movie.path, [max(0, source_time + self.analysis.offset)],
                                                  frame, width=384, labels=False)
                            if not usable_frame(frame, self.movie, source_time):
                                raise ObservationError("Skipped a black, unusable or probable title/credit frame.")
                            attempted_provider = True
                            self.loaded = True
                            description = describe_image(frame, self.model, source_time,
                                                         prompt=RETRY_PROMPT if number else PROMPT)
                            frame.replace(preview)
                            break
                        except (RuntimeError, ValueError, OSError) as error:
                            failure = str(error)
                if description is not None:
                    description["source_in"] = float(clip.start / self.movie.fps)
                    description["source_out"] = float(clip.end / self.movie.fps)
                    atomic_json(saved, description)
                elif attempted_provider:
                    provider_failures += 1
            if description is not None:
                result[clip.start] = description
            else:
                self.failed += 1
            elapsed = monotonic() - started
            remaining = elapsed * (self.total - index) / index
            progress(f"Visual AI analysis · {index}/{self.total} · {len(result)} successful · "
                     f"{self.failed} skipped/failed · source {source_time:.2f}s")
            if observe:
                observe({"image": preview.read_bytes() if description is not None and preview.is_file() else b"",
                         "description": (description.get("description") or " ".join(
                             text for key in FIELDS if key != "uncertain" for text in description.get(key, [])))
                         if description else failure,
                         "seconds": source_time, "processed": index, "total": self.total,
                         "successful": len(result), "failed": self.failed, "elapsed": elapsed, "eta": remaining})
            if provider_failures >= 4 and not result:
                break
        if cache_only:
            progress(f"Cache reuse · {len(result)}/{self.total} visual observations · no inference")
            if not result and not allow_empty:
                raise ObservationError("No compatible cached observations. Use Resume AI Review to analyze missing scenes.")
        if not result and not allow_empty:
            raise ObservationError(f"No visual evidence after {self.failed} skipped or failed scenes. "
                                   f"Last issue: {failure or 'no usable source frames'}. "
                                   "Run Test Vision AI or the Windows diagnostic, then retry.")
        return result

    def unload(self):
        if self.loaded:
            request("/api/generate", {"model": self.model, "keep_alive": 0, "stream": False}, timeout=10)
            self.loaded = False
        else:
            release_model(self.model)


def whisper_python():
    return os.getenv("MRS_WHISPER_PYTHON", "").strip()


def dialogue(movie, cache, progress, model="base", *, subtitle_file=None, ocr=False, translate=False, transcribe=True):
    from . import dialogue as evidence
    if model not in {"base", "small"}:
        raise ValueError("Choose base or small for faster-whisper.")
    path = movie.path
    sidecar = Path(subtitle_file) if subtitle_file else next((path.with_suffix(ext) for ext in (".srt", ".vtt") if path.with_suffix(ext).is_file()), None)
    legacy_stamp = ":".join([identity(path), identity(sidecar) if sidecar else "embedded", model,
                             whisper_python(), os.getenv("MRS_WHISPER_CACHE", "")])
    stamp = f"{legacy_stamp}:{ocr}:{translate}"
    saved = cache / f"dialogue-313-{sha256(stamp.encode()).hexdigest()}.json"
    progress("Dialogue transcription · checking cached subtitles or transcript")
    if saved.exists():
        return json.loads(saved.read_text(encoding="utf-8"))
    output = evidence.subtitles(path, sidecar)
    if not output["segments"] and ocr:
        try:
            ocr_output = evidence.ocr_captions(movie, progress)
            if ocr_output["segments"]:
                output = ocr_output
        except RuntimeError as error:
            progress(f"Dialogue OCR unavailable · {error}")
    if output["segments"]:
        atomic_json(saved, output)
        return output
    from .dialogue import missing_dialogue
    progress(missing_dialogue(output))
    if not (ocr or translate or subtitle_file):
        legacy = cache / f"dialogue-31-{sha256(legacy_stamp.encode()).hexdigest()}.json"
        if legacy.is_file():
            try:
                existing = json.loads(legacy.read_text(encoding="utf-8"))
                if existing.get("segments"):
                    progress("Dialogue evidence · reused a compatible earlier transcript")
                    atomic_json(saved, existing)
                    return existing
            except (ValueError, OSError):
                pass
    python = whisper_python()
    if not transcribe or not python or not movie.has_audio:
        return output
    progress(f"Dialogue transcription · faster-whisper {model} on CPU")
    worker = (Path(sys.executable).parent / "movie_review_studio" / "whisper_worker.py"
              if getattr(sys, "frozen", False) else Path(__file__).with_name("whisper_worker.py"))
    checkpoints = saved.with_suffix("")
    checkpoints.mkdir(exist_ok=True)
    completed = run_cmd([python, str(worker), str(path), model, str(checkpoints), "translate" if translate else "transcribe"], check=False)
    if completed.returncode:
        raise RuntimeError("faster-whisper failed. Check MRS_WHISPER_PYTHON and the local model setup.")
    subtitle_result = output
    output = json.loads(completed.stdout)
    if output["segments"]:
        atomic_json(saved, output)
    else:
        output = subtitle_result
    return output
