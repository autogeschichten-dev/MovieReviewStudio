from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

from movie_review_studio import adaptive, narration
from movie_review_studio.media import ffmpeg, identity, inspect
from movie_review_studio.model import Clip, timeline_entries, validate
from movie_review_studio.render import RATE, Mix, render


def clip(start, length=5, role=""):
    return Clip(start, start + length, 1, (), (), (), bool(role), shot=start, role=role)


class AdaptiveClockTests(unittest.TestCase):
    def test_fifteen_to_twenty_four_minutes_with_bookends_and_highlight(self):
        movie = SimpleNamespace(fps=Fraction(1), has_audio=True)
        analysis = SimpleNamespace(frames=10000)
        for minutes in (15, 16, 17, 18, 20, 24):
            with self.subTest(minutes=minutes):
                seconds = minutes * 60
                half = seconds // 2
                first = [clip(30 + 13 * i) for i in range(half // 5 + 1)]
                second = [clip(5500 + 13 * i) for i in range(half // 5 + 1)]
                opening, highlight, ending = clip(0, 10, "Opening"), clip(4500, 12, "Highlight"), clip(8500, 10, "Ending")
                edits = [opening, *first, highlight, *second, ending]
                recording = narration.Recording(Path("timing-fixture"), seconds * RATE, ())
                with patch.object(adaptive, "quiet_ranges", return_value=[((half - 1) * RATE, half * RATE)]):
                    picture, _, schedule, length, _ = adaptive.fit_recording(
                        recording, edits, movie, analysis, None, 5, lambda _: None)
                self.assertEqual((picture[0], picture[-1]), (opening, ending))
                self.assertIn(highlight, picture)
                self.assertEqual(adaptive.duration_totals(recording, picture, movie.fps),
                                 (seconds, 10, 12, 10, seconds + 32))
                self.assertEqual(sum(c.frames for c in picture), seconds + 32)
                self.assertEqual((length, schedule[-1][2]), (seconds * RATE, seconds * RATE))
                self.assertTrue(all(entry.segment.frames <= 5 for entry in timeline_entries(picture, movie.fps)))

    def test_unresolved_gap_and_adjacent_split_rejected_for_export(self):
        first = clip(20)
        second = replace(clip(25), shot=first.shot)
        with self.assertRaisesRegex(ValueError, "continuous source shot"):
            validate([first, second], Fraction(1), 100)
        with self.assertRaisesRegex(ValueError, "unresolved picture gaps"):
            validate([first, adaptive.placeholder(second)], Fraction(1), 100, final=True)

    def test_fractional_fps_rounds_up_to_keep_last_audio_sample(self):
        fps = Fraction(24000, 1001)
        record = narration.Recording(Path("duration-fixture"), 100001, ())
        frames = adaptive.narration_frames(record, fps)
        self.assertGreaterEqual(Fraction(frames, fps), Fraction(record.frames, RATE))
        self.assertLess(Fraction(frames - 1, fps), Fraction(record.frames, RATE))


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class AdaptiveMediaTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "source.mp4"
        ffmpeg("-f", "lavfi", "-i", "testsrc2=s=96x64:r=4:d=40", "-f", "lavfi", "-i", "sine=f=400:d=40",
               "-c:v", "libx264", "-c:a", "aac", str(self.path))
        self.movie = inspect(self.path)
        self.analysis = SimpleNamespace(frames=160, offset=0, shot_starts=(), candidates=[])
        self.progress = lambda _: None

    def test_real_short_export_exact_trim_and_optional_ending_fade(self):
        source = self.root / "voice.wav"
        ffmpeg("-f", "lavfi", "-i", "sine=f=900:d=2.75", str(source))
        part = narration.read_part(source, self.root, self.progress)
        recording = narration.merge([part], 0, False, self.root, self.progress)
        opening, middle, ending = clip(0, 8, "Opening"), clip(40, 16), clip(96, 8, "Ending")
        picture, _, schedule, length, _ = adaptive.fit_recording(
            recording, [opening, middle, ending], self.movie, self.analysis, self.root, 5, self.progress)
        self.assertEqual([c.frames for c in picture], [8, 11, 8])
        bed = narration.materialize(recording, schedule, length, self.root)
        with wave.open(str(bed), "rb") as mixed, wave.open(str(recording.path), "rb") as dry:
            self.assertEqual(mixed.readframes(mixed.getnframes()), dry.readframes(dry.getnframes()))
        out = render(self.movie, self.analysis, picture, bed, None, self.root, self.progress, mix=Mix(),
                     destination=self.root / "review.mp4", managed_voice=True, fitted_duration=True, fade_ending=True)
        self.assertAlmostEqual(inspect(out).duration, 6.75, places=3)
        pixels = ffmpeg("-i", str(out), "-an", "-vf", r"select=eq(n\,26),scale=1:1", "-f", "rawvideo",
                        "-pix_fmt", "rgb24", "pipe:1", binary=True)
        self.assertTrue(pixels and max(pixels) <= 8)

    def test_ui_locked_gap_replacement_undo_redo_and_delete_bookend(self):
        import os
        from PySide6.QtWidgets import QApplication
        from movie_review_studio.app import Studio
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        application = QApplication.instance() or QApplication([])
        app = Studio(self.root)
        self.addCleanup(app.close)
        app.movie, app.analysis, app.source_key = self.movie, self.analysis, identity(self.path)
        opening, old, replacement = clip(0, 8, "Opening"), clip(40, 12), clip(84, 12)
        app.analysis.candidates = [old, replacement]
        app.voice = self.root / "voice.wav"
        app.clips, app.highlights = [opening, old], [opening]
        app.refresh()
        app.select_edit(1)
        app.toggle_lock()
        with self.assertRaisesRegex(ValueError, "Unlock"):
            app.delete_clip()
        app.undo()
        app.select_edit(1)
        app.delete_clip()
        self.assertEqual(sum(c.frames for c in app.clips), 20)
        self.assertTrue(app.clips[1].gap)
        self.assertFalse(app.validation_report()["passed"])
        app.select_edit(1)
        with patch.object(app, "background", side_effect=lambda _, operation, apply: apply(operation(self.progress))):
            app.fill_gap()
        self.assertEqual(app.clips[1], replacement)
        app.undo()
        self.assertTrue(app.clips[1].gap)
        app.redo()
        self.assertEqual(app.clips[1], replacement)
        app.select_edit(0)
        app.delete_clip()
        self.assertEqual(app.clips, [replacement])
        self.assertTrue(app.cinema_settings["disable_Opening"])
