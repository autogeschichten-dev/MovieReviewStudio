"""Standalone worker, run by a separately configured faster-whisper Python."""
import json
import os
import sys
from pathlib import Path


def main():
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio

    path, model, checkpoint_dir, task = sys.argv[1:]
    if model not in {"base", "small"}:
        raise ValueError("Expected base or small")
    recognizer = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=6,
                             download_root=os.getenv("MRS_WHISPER_CACHE"), local_files_only=True)
    rate = 16000
    audio = decode_audio(path, sampling_rate=rate)
    language, result = None, {"provider": "faster-whisper" if task == "transcribe" else "faster-whisper (translated to English)", "language": None, "segments": []}
    step = 300 * rate
    for index, start in enumerate(range(0, len(audio), step)):
        checkpoint = Path(checkpoint_dir) / f"{index:05}.json"
        if checkpoint.is_file():
            chunk = json.loads(checkpoint.read_text(encoding="utf-8"))
        else:
            items, info = recognizer.transcribe(audio[start:start + step], language=language,
                                               vad_filter=True, beam_size=3, task=task)
            offset = start / rate
            chunk = {"language": info.language,
                     "segments": [{"start": item.start + offset, "end": item.end + offset, "text": item.text.strip()}
                                  for item in items if item.text.strip()]}
            pending = checkpoint.with_suffix(".pending")
            pending.write_text(json.dumps(chunk, ensure_ascii=True), encoding="utf-8")
            pending.replace(checkpoint)
        if chunk["segments"] and not language:
            language = chunk["language"]
        result["segments"].extend(chunk["segments"])
    result["language"] = language
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
