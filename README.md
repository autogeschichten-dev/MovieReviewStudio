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

1. **Import.** Enter the title and year, then import a local MP4, MKV or MOV from a source you have the right to use. Check the detected resolution, FPS and duration. Add an optional MP3 or WAV narration and a background track. Choose a target of 16, 17 or 18 minutes and analyze the movie.
2. **Cut.** Add two or three highlights. Enter each start time relative to the first movie frame or use **Source preview** and **Use source playhead**. Choose a 10–15 second duration and preview each highlight before generating the montage. Use the highlight selector to replace or remove a highlight.
3. Generate one montage. The timeline shows the actual output order, source and montage ranges, duration, and movie-audio mode for every video segment. Select a row to preview, replace, delete, or move a clip. Use **Play assembly** and drag either playhead to scrub. The preview shows which clip is active without changing the selected row.
4. Open **Scenes** to search analyzed footage by source time, scene number, or status. Select a candidate to preview it, append it to the timeline, or replace a selected muted clip.
5. Save the edit as an `.mrs.json` file and reopen it from **Import**. The edit references local files by path, so keep the source files in place. Older Stage 1 edits are split into short video segments when opened.
6. **Final.** Validate and export one H.264/AAC MP4, scaled to fit within 1920×1080 at the source frame rate. The exporter reports duplicate, duration and highlight validation failures before rendering.

Every video segment lasts at most five seconds. A 10–15-second highlight is split at detected shot boundaries where possible, with continuous dialogue from the source movie's first audio stream. Highlights move and replace as a group. An outer segment can be trimmed if the remaining range is at least 10 seconds. Narration and looping music advance during muted footage and pause during highlights. Audio fades at the transition between the two modes. Narration plays once and may run out or be longer than the available muted footage.

Shot analysis, preview encoding and export run in background workers. Analysis, scene thumbnails and low-resolution previews are cached. The first **Play selected** after editing may need to encode the assembly so the clip can use its actual montage audio. Qt stores the cache and `studio.log` in the app-local data directory, typically `%LOCALAPPDATA%\Movie Review Studio` on Windows. Close the app before clearing its cache. Cache use and render time depend on the length of the movie and number of edits.

## Tests

Run from Command Prompt after installation.

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The suite exercises duplicate and overlap rejection, replacement, duration checks, full timeline generation, shot-boundary cuts, fractional frame rates, caching and the narration/music/movie audio transitions. FFmpeg integration tests generate their own media and skip if FFmpeg or ffprobe is unavailable.

## Design and current limits

- The editor uses a verified constant-rate frame clock and half-open source ranges. Variable-frame-rate and rotated inputs should be converted before analysis. The first video stream must be the movie. Source preview uses Qt Multimedia and depends on decoder support.
- PySceneDetect identifies shot boundaries. OpenCV measures motion, exposure, sharpness and edge structure and computes perceptual fingerprints. Black-frame and credit filtering and near-duplicate detection are heuristic. Gradual transitions, stylized credits and heavily altered repetitions can require manual review. Short, repetitive or low-quality sources may lack enough usable material to reach 16 minutes.
- `analysis.py` indexes source frames, shot cuts, thumbnails and candidates. `model.py` owns validation, selection and segment layout. `render.py` consumes validated edit decisions and handles audio routing. Highlights use manually selected continuous source ranges.
