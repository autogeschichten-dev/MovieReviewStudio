from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, skipUnless
import shutil

from movie_review_studio import local_ai
from movie_review_studio.dialogue import coverage, missing_dialogue, parse_captions
from movie_review_studio.media import ffmpeg


class DialogueTests(TestCase):
    def test_vtt_keeps_text_and_timestamps(self):
        data = "WEBVTT\n\n00:01.000 --> 00:03.000 align:start\n<i>سلام</i> Don’t close the door.\n"
        self.assertEqual(parse_captions(data), [{"start": 1, "end": 3, "text": "سلام Don’t close the door."}])
        self.assertEqual(coverage(parse_captions(data), 4), 50)

    def test_missing_dialogue_distinguishes_subtitle_track_from_picture(self):
        self.assertIn("Embedded subtitle tracks found", missing_dialogue({"subtitle_streams": 2}))
        status = missing_dialogue({"subtitle_streams": 0})
        self.assertIn("No embedded subtitle tracks", status)
        self.assertIn("Burned-in subtitles, if present", status)
        self.assertIn("transcription", status)

    @skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_embedded_english_is_preferred_and_empty_dialogue_is_explicit(self):
        with TemporaryDirectory() as path:
            root = Path(path)
            fa, en = root / "fa.srt", root / "en.srt"
            fa.write_text("1\n00:00:00,300 --> 00:00:01,500\nسلام\n", encoding="utf-8")
            en.write_text("1\n00:00:00,400 --> 00:00:01,500\nGood morning.\n", encoding="utf-8")
            video = root / "source.mkv"
            ffmpeg("-f", "lavfi", "-i", "testsrc2=s=96x64:r=4:d=2", "-i", str(fa), "-i", str(en),
                   "-map", "0:v", "-map", "1:s", "-map", "2:s", "-c:v", "libx264", "-c:s", "srt",
                   "-metadata:s:s:0", "language=fas", "-metadata:s:s:1", "language=eng", str(video))
            result = local_ai.dialogue(SimpleNamespace(path=video, has_audio=False), root, lambda _: None)
            self.assertEqual(result["language"], "eng")
            self.assertEqual(result["segments"][0]["text"], "Good morning.")
            quiet = root / "quiet.mp4"
            ffmpeg("-f", "lavfi", "-i", "testsrc2=s=96x64:r=4:d=1", "-c:v", "libx264", str(quiet))
            result = local_ai.dialogue(SimpleNamespace(path=quiet, has_audio=False), root, lambda _: None)
            self.assertEqual(result["provider"], "none")
            self.assertEqual(result["segments"], [])
