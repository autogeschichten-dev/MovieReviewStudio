from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, skipUnless
from unittest.mock import patch
import shutil

import numpy as np

from movie_review_studio import adaptive, narration
from movie_review_studio.media import ffmpeg, frame_clock, inspect
from movie_review_studio.model import Clip, adaptive_frames, narration_frames, timeline_validation_report, validate
from movie_review_studio.render import RATE, Mix, render
from movie_review_studio.smart import AudioEvidence
from strict_fixtures import audio_group


def part(a, b):
    return Clip(a, b, 1, (), (), (), shot=a)


def group(a, b, role, spans):
    return audio_group(Clip(a, b, 1, (), (), (), highlight=True, role=role), spans)


class AudioRestorationTests(TestCase):
    def test_exact_adaptive_policy_at_reported_and_longer_durations(self):
        fps = Fraction(24000, 1001)
        for seconds in (Fraction(823133, 1000), Fraction(1030), Fraction(1130), Fraction(1260)):
            with self.subTest(seconds=seconds):
                samples = int(seconds * RATE)
                remaining = narration_frames(samples, fps)
                maximum = int(5 * fps)
                clips = []
                start = 0
                while remaining:
                    count = min(maximum, remaining)
                    clips.append(part(start, start + count))
                    start += maximum + 121
                    remaining -= count
                validate(clips, fps, start, final=True, narration_samples=samples)
                if seconds < 960 or seconds > 1080:
                    with self.assertRaisesRegex(ValueError, "16–18"):
                        validate(clips, fps, start, final=True)
                with self.assertRaisesRegex(ValueError, "picture duration"):
                    validate(clips[:-1], fps, start, final=True, narration_samples=samples)

    def test_audio_search_allows_discontinuous_source_picture_and_checks_for_missing_groups(self):
        movie = SimpleNamespace(fps=Fraction(1), has_audio=True)
        analysis = SimpleNamespace(frames=100, offset=0, candidates=[part(5, 8), part(25, 28), part(65, 70)])
        evidence = AudioEvidence((.2,) * 2000, (.2,) * 2000)
        existing = []
        omissions = []

        def choose(*args, **kwargs):
            _, _, first, last = args[:4]
            if kwargs["role"] == "Ending":
                raise ValueError("fixture rejected ending")
            return Clip(first, last, 1, (), (), (), highlight=True, role=kwargs["role"])

        with patch.object(adaptive, "fingerprint_highlight", side_effect=choose):
            result = adaptive.cinema(movie, analysis, evidence, None, lambda _: None,
                                     {"opening": 10, "ending": 10, "count": 1, "length": 12},
                                     existing, omissions=omissions)
        self.assertEqual([c.role for c in result], ["Opening", "Highlight"])
        self.assertIn("Ending: found 0 of 1", omissions[0])

    def test_pause_search_backtracks_to_keep_all_three_groups(self):
        fps = Fraction(1)
        movie = SimpleNamespace(fps=fps, has_audio=True)
        analysis = SimpleNamespace(frames=300)
        body = [part(start, start + 5) for start in (0, 30, 60, 90)]
        groups = [group(start, start + 12, "Highlight",
                        [(start, start + 5), (start + 10, start + 15), (start + 20, start + 22)])
                  for start in (120, 170, 220)]
        recording = narration.Recording(Path("fixture"), 20 * RATE, ())
        quiet = [(i * RATE, i * RATE) for i in (5, 10, 15)]
        omissions = []
        result = adaptive.insert_groups(body, [], [], list(zip(groups, (14, 15, 16))), recording,
                                        quiet, movie, analysis, None, 5, lambda _: None, [], omissions)
        self.assertEqual([clip for clip in result if clip.movie_audio], groups)
        self.assertFalse(omissions)
        self.assertEqual(adaptive.positions(result, fps)[2], [5 * RATE, 10 * RATE, 15 * RATE])

    def test_reported_adaptive_export_passes_duration_before_stale_sync_guard(self):
        import os
        from PySide6.QtWidgets import QApplication
        from movie_review_studio.app import Studio

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        application = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as folder:
            root = Path(folder)
            app = Studio(root)
            app.duration_mode = "adaptive"
            app.movie = SimpleNamespace(fps=Fraction(1), has_audio=False)
            app.voice_recording = narration.Recording(root / "test.wav", 823133 * RATE // 1000, ())
            cuts = [(i * 12, i * 12 + 5) for i in range(164)] + [(1968, 1972)]
            app.clips = [part(a, b) for a, b in cuts]
            app.analysis = SimpleNamespace(frames=3000)
            self.assertEqual(sum(c.frames for c in app.clips), 824)
            with patch.object(app, "require_movie"):
                with self.assertRaisesRegex(ValueError, "Merge & Synchronize"):
                    app.export()
            app.close()
        application.processEvents()

    def test_picture_gap_can_retain_original_audio_but_cannot_export(self):
        opening = group(0, 10, "Opening", [(0, 5), (10, 15)])
        changed = audio_group(opening, [(0, 5), (10, 15)])
        from dataclasses import replace
        changed = replace(changed, picture=(adaptive.placeholder(changed.picture[0]), changed.picture[1]))
        picture = [changed, part(20, 25)]
        validate(picture, Fraction(1), 30)
        self.assertTrue(picture[0].movie_audio)
        self.assertEqual(picture[0].frames, 10)
        with self.assertRaisesRegex(ValueError, "unresolved picture gaps"):
            validate(picture, Fraction(1), 30, final=True, narration_samples=5 * RATE)


@skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class ActualAudioExportTests(TestCase):
    def test_export_keeps_three_original_signals_and_narration_without_overlap(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            movie = root / "source.mp4"
            original = r"aevalsrc=0.2*sin(2*PI*if(lt(t\,20)\,310\,if(lt(t\,55)\,510\,710))*t):s=48000:d=100"
            ffmpeg("-f", "lavfi", "-i", "testsrc2=s=96x64:r=1:d=100", "-f", "lavfi", "-i", original,
                   "-c:v", "libx264", "-c:a", "aac", str(movie))
            voice = root / "voice.wav"
            narration_signal = r"aevalsrc=0.18*sin(2*PI*if(lt(t\,5)\,910\,1110)*t):s=48000:d=10"
            ffmpeg("-f", "lavfi", "-i", narration_signal, str(voice))
            picture = [group(0, 10, "Opening", [(0, 5), (10, 15)]), part(22, 27),
                       group(30, 42, "Highlight", [(34, 38), (45, 49), (56, 60)]),
                       part(66, 71), group(60, 70, "Ending", [(80, 85), (94, 99)])]
            source = inspect(movie)
            analysis = SimpleNamespace(frames=100, offset=0)
            expected = adaptive_frames(picture, source.fps, 10 * RATE)
            report = timeline_validation_report(picture, source.fps, analysis.frames, narration_samples=10 * RATE)
            self.assertTrue(report["passed"])
            self.assertFalse(report["violations"])
            self.assertEqual(report["original_audio_visual_exceptions"], 0)
            export = render(source, analysis, picture, voice, None, root, lambda _: None,
                            destination=root / "review.mp4", mix=Mix(), managed_voice=True, narration_samples=10 * RATE)
            self.assertEqual(frame_clock(inspect(export))[0], expected)
            self.assertEqual(inspect(export).duration, 42)
            raw = ffmpeg("-i", str(export), "-map", "0:a:0", "-ar", str(RATE), "-ac", "1", "-f", "s16le", "pipe:1", binary=True)
            pcm = np.frombuffer(raw, dtype="<i2")
            for second, frequency in ((2, 310), (7, 310), (12, 910), (16, 510),
                                      (20, 510), (24, 510), (29, 1110), (34, 710), (39, 710)):
                samples = pcm[second * RATE:(second + 1) * RATE].astype(float)
                magnitude = np.abs(np.fft.rfft(samples))
                self.assertEqual(np.argmax(magnitude), frequency)
                others = [magnitude[f] for f in (310, 510, 710, 910, 1110) if f != frequency]
                self.assertLess(max(others), magnitude[frequency] * .03)
