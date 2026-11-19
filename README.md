# Movie Review Studio

Movie Review Studio is a standalone Windows desktop editor for making one 16–18 minute review montage from a local movie. It uses Python, PySide6, FFmpeg, PySceneDetect and OpenCV. It has no dependency on Movie Montage Studio.

## Portable Windows x64 package

Download the ZIP from [GitHub Releases](https://github.com/autogeschichten-dev/MovieReviewStudio/releases), extract the entire archive, and double-click `RUN_WINDOWS.bat` or `MovieReviewStudio.exe`. Keep the extracted folder together. The package includes Python, Qt, FFmpeg, and FFprobe.

The **Windows x64 package** GitHub Actions workflow builds the portable ZIP, runs the test suite, and smoke-tests the extracted editor. A push of a `stage1-windows-v*` tag publishes the checked ZIP as a GitHub Release asset.

## Windows setup from source

1. Install 64-bit [Python 3.12](https://www.python.org/downloads/windows/) with the Windows launcher (`py`).
2. Install a Windows [FFmpeg build](https://ffmpeg.org/download.html#build-windows) with `libx264` and AAC. Add its `bin` directory to PATH, then open a new terminal and check `ffmpeg -version` and `ffprobe -version`.
3. Run `install.bat` from the checkout. This installs the pinned Python packages into `.venv`.
4. Run `run.bat` from the checkout to launch the editor.

## Create a review

1. **Import.** Enter the title and year, then import a local MP4, MKV or MOV from a source you have the right to use. Check the detected resolution, FPS and duration. Add an optional MP3 or WAV narration and a background track. Choose a target of 16, 17 or 18 minutes and analyze the movie.
2. **Cut.** Add two or three highlights. Enter each start time relative to the first movie frame or use **Source preview** and **Use source playhead**. Choose a 10–15 second duration and preview each highlight before generating the montage. Use the highlight selector to replace or remove a highlight.
3. Generate one montage. Click a row to select it and use **Play selected**, **Replace selected** or **Delete selected**. The gold rows and timeline segments indicate original-audio highlights. Play the assembly and drag either playhead to scrub. Save the edit as an `.mrs.json` file and reopen it from **Import**. The edit references local files by path, so keep the source files in place.
4. **Final.** Validate and export one H.264/AAC MP4, scaled to fit within 1920×1080 at the source frame rate. The exporter reports duplicate, duration and highlight validation failures before rendering.

Regular clips last at most five seconds. They have no movie audio. Narration and looping music advance during regular footage and pause during highlights. Both fade at transitions into and out of highlights. Highlights use the source movie's first audio stream, so choose a source with the intended audio track first. Narration plays once and may run out or be longer than the available regular footage.

Shot analysis, preview encoding and export run in background workers. Analysis and low-resolution previews are cached by source identity and edit ranges. Qt stores the cache and `studio.log` in the app-local data directory, typically `%LOCALAPPDATA%\Movie Review Studio` on Windows. Close the app before clearing its cache. Cache use and render time depend on the length of the movie and number of edits.

## Tests

Run from Command Prompt after installation.

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The suite exercises duplicate and overlap rejection, replacement, duration checks, full timeline generation, shot-boundary cuts, fractional frame rates, caching and the narration/music/movie audio transitions. FFmpeg integration tests generate their own media and skip if FFmpeg or ffprobe is unavailable.

## Design and current limits

- Stage 1 uses a verified constant-rate frame clock and half-open source ranges. Variable-frame-rate and rotated inputs should be converted before analysis. The first video stream must be the movie. Source preview uses Qt Multimedia and depends on decoder support.
- PySceneDetect identifies shot boundaries. OpenCV measures motion, exposure, sharpness and edge structure and computes perceptual fingerprints. Black-frame and credit filtering and near-duplicate detection are heuristic. Gradual transitions, stylized credits and heavily altered repetitions can require manual review. Short, repetitive or low-quality sources may lack enough usable material to reach 16 minutes.
- `analysis.py` indexes source frames and candidates. `model.py` owns validation and selection, and accepts a `ClipRanker` for future narration-aware ranking. The `HighlightRecommender` protocol allows future systems to propose frame ranges for manual review. `render.py` consumes validated edit decisions and handles audio routing. Stage 1 uses manual highlight selection.
