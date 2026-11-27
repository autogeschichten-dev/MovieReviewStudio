from fractions import Fraction
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

from movie_review_studio import director
from movie_review_studio.analysis import Analysis
from movie_review_studio.media import ffmpeg, inspect
from movie_review_studio.model import Clip


class DirectorValidationTests(unittest.TestCase):
    def test_rejects_non_local_http_provider(self):
        from unittest.mock import patch
        with patch.dict("os.environ", {"MRS_AI_URL": "http://example.com/v1", "MRS_AI_MODEL": "vision"}):
            with self.assertRaisesRegex(ValueError, "cloud HTTPS"):
                director.VisionProvider.from_environment()

    def test_transcript_alignment_matches_accented_sections_in_order(self):
        script = [director.ScriptSection("part-1", "La tradición sigue viva.", ()),
                  director.ScriptSection("part-2", "Ella habló con su familia.", ())]
        cue = [{"start": 0.2, "end": 1.9, "text": "La tradición sigue viva"},
               {"start": 3.0, "end": 4.2, "text": "Ella habló con su familia"}]
        self.assertEqual(director.match_recording(script, cue), {"part-1": (.2, 1.9), "part-2": (3, 4.2)})
        with self.assertRaisesRegex(ValueError, "match"):
            director.match_recording(script, list(reversed(cue)))


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Requires FFmpeg")
class RecapLibraryTests(unittest.TestCase):
    def test_short_real_upstream_evidence_and_frame_exact_timeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            ffmpeg("-f", "lavfi", "-i", "testsrc2=s=160x90:r=12:d=11",
                   "-f", "lavfi", "-i", "sine=frequency=340:duration=11",
                   "-c:v", "libx264", "-c:a", "aac", "-shortest", str(source))
            source.with_suffix(".srt").write_text(
                "1\n00:00:00,200 --> 00:00:01,800\nLa familia regresa.\n", encoding="utf-8")
            movie = inspect(source)
            metadata = director.probe.probe_sources([source])
            self.assertTrue(metadata["sources"][0]["has_video"])
            evidence = director.read_transcript(movie, root, lambda _: None)
            self.assertEqual(evidence[0]["text"], "La familia regresa.")
            thumb = root / "frame.jpg"
            director.frames.extract_frames(source, [1.25], thumb, width=96, labels=False)
            self.assertTrue(thumb.read_bytes().startswith(b"\xff\xd8"))
            clips = [Clip(0, 36, 0, (), (), ()), Clip(96, 120, 0, (), (), (), original=True)]
            analysis = Analysis(132, 0, clips, ())
            section = director.ScriptSection("part-1", "La familia regresa.", ((0, 36, "Caption at 0.2s"),))
            doc = director.timeline_document(movie, clips, [section])
            self.assertEqual(director.import_timeline(doc, movie, analysis, min_gap=1), clips)
            self.assertEqual(doc["meta"]["script_sections"][0]["placements"][0]["montage_out"], 3)
            report = director.recap_render.validate_timeline_physical(doc, strict=True)
            self.assertEqual(report["duration_s"], 5)
            doc["video"][0]["out"] = 5.5
            with self.assertRaisesRegex(ValueError, "fragment"):
                director.import_timeline(doc, movie, analysis, min_gap=1)

    def test_mock_vision_contract_rejects_invented_shot_ids(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            movie = SimpleNamespace(path=root / "source.mp4", fps=Fraction(1), width=160, height=90)
            movie.path.write_bytes(b"fixture")
            candidates = [Clip(start, start + 5, 0, (), (), (), shot=start) for start in range(0, 1300, 5)]
            analysis = Analysis(1300, 0, candidates)
            provider = SimpleNamespace(complete=lambda *_: {"text": "Draft", "shots": [{"id": "invented", "observation": "None"}]})
            def image(*args, **kwargs):
                args[2].write_bytes(b"fixture")
            with patch.object(director.probe, "probe_sources", return_value={"sources": [{"has_video": True}]}), \
                 patch.object(director, "read_transcript", return_value=[]), \
                 patch.object(director.frames, "extract_frames", side_effect=image):
                with self.assertRaisesRegex(ValueError, "unknown shot"):
                    director.plan(movie, analysis, [], provider, director.MODES[0], "English", 1050, "", root, lambda _: None)
