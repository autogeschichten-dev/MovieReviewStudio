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

## V3.1 local AI workflow

Start Ollama and keep `moondream:latest` installed. In **Import**, choose a movie and open **Analyze Movie / Generate AI Review**. The dialog lists installed Ollama models and shows connection status. Start **Analyze Movie** to cache a chronological sample of actual source frames and local dialogue cues. Then choose a separate text model and click **Generate AI Review**. Select Cultural Stories, Life Lessons, Psychological Analysis, or Custom Script. Model observations may be mistaken. Review visible evidence and uncertainty before approving the script. **Fast Montage** requires neither model.

The visual stage sends one JPEG at a time to `http://127.0.0.1:11434/api/generate`, with a 2,048-token context and an indexed, timestamped cache. Moondream is unloaded before the planner runs. A local planner on the same Ollama service is unloaded before vision and after planning. Choose an installed text model that fits your machine and can follow JSON instructions. Select it in the dialog or set `MRS_TEXT_MODEL`. The default text API is `http://127.0.0.1:11434/v1`. An optional compatible remote planner uses `MRS_TEXT_URL` and `MRS_TEXT_KEY`, and needs confirmation before sending descriptions, dialogue excerpts or custom script text. Movie frames stay on the machine. The V3 vision variables above describe the legacy provider interface.

For an RTX 3050 with 4 GB VRAM, run vision, transcription and planning sequentially. The initial visual pass can sample up to 640 stills, so measure seconds per image before a full run. A small text model with a larger supported context may require CPU offload or a separately configured planner. Timestamps and uncertainty are preserved, but relevance and factual accuracy still require review. Completed scenes, transcript chunks, and planner responses are cached. A cancelled request may need to finish before the app can unload the model.

### Local faster-whisper on Windows

Subtitles take priority. To enable local speech recognition, use a separate Python 3.12 environment with faster-whisper installed. The app uses CPU INT8, six CPU threads, automatic language detection and five-minute checkpoints. Models are pre-downloaded by the user. The app opens them locally only. Example in PowerShell:

```powershell
py -3.12 -m venv .whisper-venv
.\.whisper-venv\Scripts\python.exe -m pip install faster-whisper
.\.whisper-venv\Scripts\python.exe -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"
$env:MRS_WHISPER_PYTHON = (Resolve-Path .\.whisper-venv\Scripts\python.exe).Path
.\RUN_WINDOWS.bat
```

Repeat the pre-download with `small` if you select that model. `MRS_WHISPER_CACHE` optionally points at the Hugging Face download cache. There is no automatic model download during editing.

### Measure this PC

Double-click `RUN_AI_DIAGNOSTIC.bat` in the portable ZIP for a four-second red-and-blue fixture. The report includes the actual images' byte counts and hashes, raw observations, known-color checks, response time, Ollama VRAM, sampled NVIDIA memory and system RAM, and total elapsed time. Then drag an authorized short video with audio onto the batch file. Repeat with full-length authorized footage. The default is 12 sampled stills per supplied movie, configurable with `MRS_DIAG_SAMPLES` up to 640. The optional video also checks subtitle or transcript timing and reports faster-whisper throughput when configured. Inspect scene descriptions for false details. The synthetic color check measures only recognition of two colors. The report opens in Notepad from `%TEMP%\MovieReviewStudio-AI-Diagnostic.txt`.

### V3.1.1 AI Settings and troubleshooting

Use **AI Settings** from Import or Smart Editor. Choose the detected Moondream and Qwen models, then use **Test Vision AI**, **Test Text AI**, or **Test All Connections**. Choices are saved on this PC. Local operation with installed Ollama models requires no environment variables. The settings dialog displays available RAM and NVIDIA adapter information. GPU details depend on access to `nvidia-smi`.

Vision uses a short descriptive prompt and accepts ordinary text, known caption keys, or the earlier structured response. A valid caption is kept as a model report. Explicitly tentative language is separated and omitted from confirmed facts. Black, low-detail and probable title/credit frames are skipped using local heuristics. A failed frame gets one retry with a shorter prompt and another timestamp from the same scene. Successful descriptions remain cached. Failed scenes are counted, and the planner uses available evidence. If the provider repeatedly fails or the remaining footage cannot support a grounded review, the app reports the problem instead of filling the gap with invented observations.

The live analysis panel displays the latest frame and description, source time, attempted scene count, successful and skipped/failed counts, elapsed time and estimated time remaining. After extracting the portable release, double-click `RUN_AI_DIAGNOSTIC.bat` to test two known-color images. The local report includes request options, image byte counts and hashes, HTTP status, and the **raw Ollama response**. It omits the base64 image text for readability. The color check is a basic recognition check, and the raw descriptions let you review false additions.

### V3.1.2 planning recovery

For a movie already analyzed in V3.1.1, reopen the same source path and choose the same Moondream model. Open **AI Director → Retry Planning Only** to reuse the recorded observations without running Moondream. **Resume AI Review** can fill missing observations. A paused or failed run offers **Retry failed chapter**, **Resume AI Review**, and **Open diagnostic report**. Earlier valid summaries and chapters are reused for the same source, planner and settings.

The planner summarizes chronological evidence batches, combines them into an outline, and writes and validates each narration chapter against a canonical shot list. Local Ollama planning uses an explicit JSON schema and candidate-ID enum. Numeric indexes and IDs from other chapters are rejected. A bounded chapter retry keeps valid references. If needed, deterministic fallback can choose a verified shot from the same chapter when its description shares evidence terms with the narration. These clips are labeled in the script evidence and diagnostic report. If related footage is insufficient, planning pauses for review.

The diagnostic is written locally as JSON Lines. It includes the candidate IDs, accepted and rejected references, retry status, and raw HTTP replies for local Ollama planning. It may contain excerpts of subtitles, scene descriptions or private script text. Review the file before sharing it. **Test All Connections** and `RUN_AI_DIAGNOSTIC.bat` check both the Moondream image path and Qwen's constrained JSON shot response when those models are installed.

### V3.1.3 evidence and repetition recovery

**AI Director → Retry Planning Only** retains valid visual observations from earlier releases when the source file path and selected vision model are unchanged. The planner packs complete descriptions into estimated token budgets, groups consecutive duplicate descriptions with their timestamps and IDs, and filters explicit unsupported film or character naming. Cached summaries from the older truncated evidence format are incompatible and are regenerated. Compatible summaries and split-batch checkpoints resume from the failed stage.

Qwen planning reports prompt/output token counts where Ollama provides them. It detects repeated phrases, retries with stronger repetition settings, and splits evidence batches after repeated failure. For advanced tuning, `MRS_REPEAT_PENALTY` defaults to 1.15 and `MRS_REPEAT_LAST_N` defaults to 128. Local JSON and shot IDs are still validated. The script distinguishes model observations from stronger evidence such as subtitle text and identifies fallback selections.

In AI Director, use **Subtitle file** to choose an SRT or VTT explicitly. The default searches a same-name sidecar and then all supported embedded text tracks, preferring English when language metadata is available. If neither works, configure **faster-whisper** as described above. Its multilingual base and small models can recognize Persian and optionally translate recognized speech to English. Image-only subtitle streams and burned-in text are separate from embedded text tracks. **Try local OCR** samples the lower part of the image with Tesseract English data from PATH. OCR results and timing are approximate and may include other on-screen text. The dialogue status shows the source, cue count, and fraction of source time covered. Zero cues is reported explicitly.

Run `RUN_AI_DIAGNOSTIC.bat` on Windows for the two known-image checks and a native Qwen summary test. The report includes raw Ollama replies and repetition/length errors where available. Title and character filters are heuristic, and the final essay still needs human evidence review.

### Source separation and validation report

Fast Montage, AI Review, narration synchronization, preview and export validate every rendered picture fragment. The source separation setting has a five-second floor. The validator checks source shot identity, touching ranges, source overlap, visually similar neighbors, and individual durations. It checks internal picture cuts in opening, highlight, and ending groups as well as joins across groups or chapters.

Original-audio groups use a separate continuous audio range. Automatic selection starts with an in-sync source shot when available and fills the remainder with eligible nearby cutaways. When picture cannot be placed safely, the group is skipped or needs a shorter range. Automated cutaway screening uses sampled frames and cannot guarantee semantic relevance or detect every visible speaker.

Old projects with contiguous highlight picture open for repair. Replace or remove those groups before playing or exporting the assembly. **Save timeline validation report…** lists unresolved fragments, source shots, source times, signed separation, audio mode and the repair status. Export saves the same audit next to the MP4.

### Multi-part voice-over

Open **Voice-over Manager** from Import or the Smart Editor inspector. Use the three default slots or add more, then drop WAV/MP3 files in playback order. Preview individual parts, reorder with the arrows, or replace or delete any recording. **Play All** and **Export WAV / MP3** use the processed combination. The manager uses 48 kHz stereo audio, per-part loudness matching with peak headroom, optional conservative edge-silence trimming, and an adjustable pause between parts.

Choose **Merge & Synchronize** after assembling the picture. The manager displays the measured combined duration, the target, and the narration capacity available outside original-audio cuts. To shorten the picture explicitly, select **Remove surplus picture clips to fit narration**. This keeps whole protected clips and can leave a short amount of excess picture. If the recording needs more relevant footage or a safe pause around a highlight, the manager reports the conflict and leaves the edit available for adjustment.

For script timing, use an SRT/VTT timed to the combined recording, a same-name sidecar for every part, or an existing configured local transcription provider. Strong subtitle and scene-description matches can refine timing within a script paragraph. Without timing evidence, synchronization follows the existing picture order and verified audio pauses. Keep the original audio files with the saved project so derived audio can be rebuilt.

### Adaptive duration and picture gaps

The Smart Editor defaults to **Adaptive to Narration** when narration parts are imported. The calculated duration is the measured merged audio plus the opening, intermediate original-audio groups, and ending. MP4 picture duration is rounded up to a complete source-rate frame, preserving the final audio sample. Select **Fixed Duration** to use the established target workflow.

**Original audio…** sets defaults for generation, including a 10-second opening and ending, two 12-second intermediate highlights, and an optional fade to black. Use the inspector and source browser to replace or trim groups already on the timeline. A deleted opening or ending stays disabled for the project until explicitly re-enabled. Cinematic cuts require usable source audio and credible source boundaries. If an intermediate cut cannot align to a verified narration pause, move it or choose a different pause.

Deleting or trimming narrated picture leaves an explicit gap so later narration and cuts stay in place. Use **Auto fill gap** or select a source candidate and choose **Replace clip**. Splitting narrated picture retains one source fragment and makes a gap for the other so adjacent fragments of the same source shot cannot be exported. Gaps remain visible and block export. Use the lock icon to protect a clip, Delete with timeline focus, and Ctrl+Z / Ctrl+Y to undo or redo.

### Planning recovery

With the same source, settings, and models, **Retry failed chapter** reads cached observations without starting visual inference. **Resume AI Review** checks those caches first and can analyze additional footage when needed. If timeline validation or writing fails after chapter selection, the validated source ranges and script checkpoint are available for retry. **Open diagnostic report** opens the per-source trace in the application cache. Python tracebacks are also written to `studio.log`.

For shortages of up to five seconds, recovery first looks for an unused portion of an already grounded shot. The retained range contains the observed frame and passes the same overlap, source-gap, and visual-cut checks as the rest of the edit.

### Narration-first highlight timing

In Adaptive to Narration, **Merge & Synchronize** sizes the picture to the processed recording first. It looks for verified quiet frames near each intermediate original-audio cut and also considers pauses between imported parts. Timestamped narration cues, when available, exclude frames inside speech. Opening and ending audio remain outside the narration. An unlocked intermediate highlight with no compatible pause is skipped with a message. Undo restores the earlier timeline. Locked cuts stay protected.

If the picture is too short, synchronization can add unused cached alternatives approved by the existing script. Without a linked script it can extend the existing source order with eligible cached scenes. Every resulting cut is validated before the edit is applied.

### Adaptive export and original-audio restoration

Adaptive export uses the processed narration sample count, rounded up to a picture frame, plus enabled original-audio frames. A stale sync state prompts for synchronization. **Merge & Synchronize** looks for missing enabled audio groups using cached shot analysis. Before applying an edit that would omit an enabled section, the editor lists the missing groups and asks whether to proceed.

Deleting an individual picture fragment from an independent original-audio group creates a gap and preserves the audio. Use **Auto fill gap** or choose a replacement in the source browser. Remove the entire audio interval explicitly in **Manual highlights**. Picture gaps block export until repaired.
