# Movie Review Studio

Movie Review Studio is a standalone Windows desktop editor for making one 16–18 minute review montage from a local movie. It uses Python, PySide6, FFmpeg, PySceneDetect and OpenCV. It has no dependency on Movie Montage Studio.

## Portable Windows x64 package

Download the ZIP from [GitHub Releases](https://github.com/autogeschichten-dev/MovieReviewStudio/releases), extract the entire archive, and double-click `RUN_WINDOWS.bat` or `MovieReviewStudio.exe`. Keep the extracted folder together. The package includes Python, Qt, FFmpeg, and FFprobe.

The complete project source ZIP also has a top-level `RUN_WINDOWS.bat`. Extract the ZIP to a writable folder and double-click that file. The first launch needs Internet access to download Python, FFmpeg, FFprobe, and any missing Python dependencies into `.windows-runtime`. Installation is local to the extracted folder and requires no administrator access or manual Python setup.

The **Windows x64 package** GitHub Actions workflow tests the first-launch installer, runs the test suite, and smoke-tests the extracted portable editor. A push of a `stage2-windows-v*` tag publishes the checked portable ZIP with the full project source as a GitHub Release asset.

## Windows setup from source

1. Install 64-bit [Python 3.12](https://www.python.org/downloads/windows/) with the Windows launcher (`py`).
2. Install a Windows [FFmpeg build](https://ffmpeg.org/download.html#build-windows) with `libx264` and AAC. Add its `bin` directory to PATH, then open a new terminal and check `ffmpeg -version` and `ffprobe -version`.
3. Run `install.bat` from the checkout. This installs the pinned Python packages into `.venv`.
4. Run `run.bat` from the checkout to launch the editor.

## Create a review

1. In **Import**, select a local movie you have permission to edit. Pick a target from 17:00 to 18:00. The default is 17:30.
2. Click **Auto Generate Movie Review**. The editor analyzes the source and arranges usable shots in source order. Audio evidence can suggest up to three continuous original-audio highlight groups. Generation can use entirely muted footage when a suitable highlight is unavailable.
3. In **Smart Editor**, click a timeline segment to inspect its source frame and metadata. Use **Play review** for the encoded assembly, or **Preview** in the inspector for the selected clip with its assembly audio. Drag clips to reorder them, drag the ruler to scrub, and use **Undo** and **Redo** to revisit edits. Highlights move as continuous groups even though their internal video segments are visible.
4. Optionally import narration and music using the inspector. Blue video denotes the narration track or its placeholder, orange uses movie audio, and purple marks clips labeled Intro or Outro. Green shows background music. The three audio tracks show placement and real waveform peaks. Narration pauses during movie dialogue, and music is reduced during narration and original-audio sections.
5. Expand **Scenes** for thumbnails, source-time search, filters, and recommended replacements. Select a scene to preview it, then use **Add scene** or **Replace clip**. Open **Manual highlights** to define, preview, replace, or remove a continuous highlight.
6. Use **Save project** to save the edit as an `.mrs.json` file. Older project files can still be opened. Projects reference media by path and save clip order, audio modes, and volume. Open **Export** to render H.264/AAC MP4 at up to 1920×1080.

Each video segment is at most five seconds. A highlight keeps its source dialogue in order and uses short fades at the group's audio edges. Shot analysis, preview encoding, and export run in workers. Use **Cancel** to interrupt processing. The first montage or selected-clip preview may take time to encode. Qt stores the cache and `studio.log` in the app-local data directory, typically `%LOCALAPPDATA%\Movie Review Studio` on Windows.

### Optional local dialogue evidence

The editor uses timing from a same-name SRT/VTT file or an embedded text subtitle stream when available. For local speech transcription, set `MRS_WHISPER_CLI` and `MRS_WHISPER_MODEL` to your local whisper.cpp executable and model paths. Choose a multilingual model for automatic language detection. The app uses the resulting timestamps as dialogue cues. Without them, it uses audio activity and visual cues. These signals are heuristic and do not establish what a character says, how they feel, or the cultural meaning of a scene.

## Tests

Run from Command Prompt after installation.

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The suite covers automatic generation, chronological placement, duplicate and overlap rejection, duration checks, shot-boundary cuts, fractional frame rates, cache use, project compatibility, undo/redo, and audio transitions. FFmpeg integration tests generate their own media and skip if FFmpeg or ffprobe is unavailable.

## Design and current limits

- The editor uses a verified constant-rate frame clock and half-open source ranges. Variable-frame-rate and rotated inputs should be converted before analysis. The first video stream must be the movie. Source preview uses Qt Multimedia and depends on decoder support.
- PySceneDetect identifies shot boundaries. OpenCV measures motion, exposure, sharpness and edge structure and computes perceptual fingerprints. Black-frame and credit filtering and near-duplicate detection are heuristic. Gradual transitions, stylized credits and heavily altered repetitions can require manual review. Short, repetitive or low-quality sources may lack enough usable material to reach 16 minutes.
- `analysis.py` indexes source frames, shot cuts, thumbnails and candidates. `model.py` owns validation, selection and segment layout. `render.py` consumes validated edit decisions and handles audio routing. Suggested and manual highlights both use continuous source ranges.

## V3 AI Director

**Fast Montage** remains available offline with the same FFmpeg editor and export workflow. **AI Director…** adds a script-driven workflow with Cultural Stories + Life Lessons as the default, separate Cultural Stories, Life Lessons and Psychological Analysis modes, and Custom Script for UTF-8 TXT or pasted text. Choose English, German or Spanish before generating a draft. Inspect the draft against its source evidence, apply edits, then approve the script. The script picker seeks the montage. When changing a linked clip, use **Link clip** and enter the evidence for the new association. Editing footage or script text invalidates a rendered narration bed; make or align the voice again after approving the edit.

### Configure the vision model

Set the following environment variables before launching Movie Review Studio.

| Setting | Purpose |
| --- | --- |
| `MRS_AI_URL` | OpenAI-compatible API base URL ending in `/v1`, such as `http://127.0.0.1:11434/v1` for a compatible local server. Remote endpoints must use HTTPS. |
| `MRS_AI_MODEL` | Model identifier for a vision model that supports inline image data, chat completions, and JSON response format. |
| `MRS_AI_KEY` | Bearer key if the endpoint requires one. Required for cloud endpoints. |

Loopback requests bypass proxy settings. Remote requests require confirmation for each run. The request contains sampled stills, timestamps, local subtitle/transcript excerpts and any custom script. The app does not send original movie or audio files. Review provider privacy, pricing, commercial-use and voice terms before configuring a cloud service. The app cannot estimate prices for arbitrary endpoints. A full review typically requests 12–32 chapters with at most 40 images per chapter. The cache keeps local stills, transcripts and responses, allowing a retry to reuse completed responses from an identical request. Allow disk space for source, previews, exports, media cache, and any separately installed local model. The first pass can take many minutes, depending on media length and model throughput. An in-flight model request has a 90-second connection/read timeout, so cancellation can wait for that request to finish.

Movie Review Studio validates evidence IDs against locally extracted frames and verifies source ranges and timeline durations. A model's interpretation still needs human review. If the model finds too little relevant footage, try a revised script or use Fast Montage. Vision weights and API access are supplied separately.

### Narration and quality checks

- **Make voice** uses Video Recap Skills' explicitly offline TTS engine. Install eSpeak NG and put `espeak-ng` on PATH on Windows or Linux. The defaults are the English, German and Spanish voices. Use `MRS_TTS_VOICE_EN`, `MRS_TTS_VOICE_DE` and `MRS_TTS_VOICE_ES` to override voice names. On macOS the upstream engine can use `say`, which needs an installed voice name. No paid speech service is enabled.
- You can import WAV/MP3 without transcription. To time an approved script against that recording, choose **Align imported narration…** with a matching same-name SRT/VTT or local Whisper transcript. For whisper.cpp use `whisper-cli` on PATH and set `WHISPER_MODEL_PATH` to a local model; the existing V2 waveform workflow has its own optional Whisper configuration. Alignment refuses weak text matches and speech that exceeds its footage, without changing the current edit.
- **Export timeline JSON…** exchanges timestamped picture, whole-clip audio modes and script metadata using the upstream v3 format. Import supports this app's single-source editing subset. Multi-source edits, overlays, crop directives and externally authored audio routing require the upstream tools. **Check exported video…** writes an upstream physical check report with any flags for review. Existing project versions still open; V3 stores scripts in project version 4.

See [third-party notices](THIRD_PARTY_NOTICES.md).
