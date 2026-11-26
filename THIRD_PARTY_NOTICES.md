# Third-party notices

Movie Review Studio V3 incorporates scripts from Video Recap Skills, copyright (c) 2026 PiteChen, under the MIT License. The included scripts are pinned to commit `41597bf23c4b0782286705ad2726a690406d7165`. See [the upstream repository](https://github.com/zenstory-ai/video-recap-skills/tree/41597bf23c4b0782286705ad2726a690406d7165) and the [complete MIT license and notice](movie_review_studio/recap_upstream/LICENSE).

The vendored scripts have package-relative imports and use the desktop application's bundled media paths, process cancellation, and Windows process flags. V3.1 also enforces the declared frame rate after composing the upstream timeline. The toolkit's physical timeline validator and quality report complement Movie Review Studio's renderer.

The portable ZIP includes the FFmpeg/x264 license and corresponding source in `FFmpeg-Source`. Python package notices are present under `_internal`. eSpeak NG and Whisper are optional separately installed tools. Their voices, models, and service terms are separate from the Video Recap Skills MIT license.

The multi-part narration package also includes LAME 3.100 for MP3 encoding. LAME's corresponding source archive and LGPL notice are in `FFmpeg-Source`, alongside the pinned build script.
