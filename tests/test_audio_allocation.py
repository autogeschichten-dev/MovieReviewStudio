"""Regression against source scale and missing dialogue in the reported failure."""
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, skipUnless
from unittest.mock import patch
import json
import os
import shutil
import wave

import numpy as np

from movie_review_studio import adaptive, planning
from movie_review_studio.media import ffmpeg, identity, inspect
from movie_review_studio.model import Clip, timeline_entries
from movie_review_studio.render import RATE
from movie_review_studio.smart import AudioEvidence
import test_mandatory_audio_pipeline as pipeline


class AllocationUnitTests(TestCase):
    def test_later_slot_causes_earlier_reservation_to_backtrack(self):
        movie = SimpleNamespace(fps=Fraction(1), has_audio=True)
        shots = [Clip(start, start + 4, 1, (), (), (), shot=start) for start in (16, 34, 58, 82)]
        analysis = SimpleNamespace(frames=100, offset=0, candidates=shots)
        evidence = AudioEvidence((.2,) * 2000, (.2,) * 2000)
        audit = []

        def select(movie, analysis, start, end, **kwargs):
            if kwargs["role"] == "Ending" and (
                    start != 82 or any(c.start == 16 for c in kwargs["reserved"])):
                raise ValueError("candidate chain conflicts")
            return Clip(start, end, 1, (), (), (), highlight=True, role=kwargs["role"])

        with patch.object(adaptive, "fingerprint_highlight", side_effect=select):
            chosen = adaptive.cinema(movie, analysis, evidence, None, lambda _: None,
                                     {"opening": 10, "count": 0, "length": 12, "ending": 10}, audit=audit)
        self.assertEqual(len(chosen), 2)
        self.assertNotEqual(chosen[0].start, 16)
        self.assertGreater(audit[0]["counts"]["backtracked"], 0)
        self.assertGreater(audit[1]["counts"]["no_complete_picture_chain"], 0)


    def test_original_picture_reservation_replaces_unlocked_narration_with_grounded_alternate(self):
        from movie_review_studio.director import ScriptSection
        from movie_review_studio.narration import Recording
        from strict_fixtures import audio_group

        movie = SimpleNamespace(fps=Fraction(1), has_audio=True)
        occupied = Clip(0, 5, 1, (), (), (), shot=0)
        alternate = Clip(40, 45, 1, (), (), (), shot=40)
        opening = audio_group(Clip(70, 80, 1, (), (), (), highlight=True, role="Opening"),
                              [(0, 5), (16, 21)])
        analysis = SimpleNamespace(frames=100, candidates=[occupied, alternate])
        script = [ScriptSection("part-1", "A local detail.", ((0, 5, "detail"),), ((40, 45),))]
        with patch.object(adaptive, "quiet_ranges", return_value=[]):
            result, _, schedule, length, _ = adaptive.fit_recording(
                Recording(Path("fixture"), 5 * RATE, ()), [occupied], movie, analysis, None, 5, lambda _: None,
                script=script, restored=[opening])
        self.assertEqual(result, [opening, alternate])
        self.assertEqual(schedule, ((0, 0, 5 * RATE),))
        self.assertEqual(length, 5 * RATE)


@skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class AllocationScaleTests(TestCase):
    def test_three_part_narration_and_original_audio_at_reported_candidate_scale(self):
        from PySide6.QtWidgets import QApplication
        from movie_review_studio.app import Studio
        from movie_review_studio.analysis import analyze

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        application = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "large-synthetic.mp4"
            pipeline.MandatoryAudioPipelineTests.make_movie(root, source, shots=1324)
            movie = inspect(source)
            analysis = analyze(movie, root, lambda _: None)
            self.assertEqual(len(analysis.candidates), 1324)
            observations = {c.start: {"source_in": float(c.start / movie.fps),
                "source_out": float(c.end / movie.fps), "timestamp": float(c.start / movie.fps),
                "description": "A synthetic geometric pattern occupies the frame."}
                for c in analysis.candidates[:628]}
            self.assertEqual(len(planning.registry(analysis.candidates, observations, movie, analysis.frames)), 628)
            studio = Studio(root)
            self.addCleanup(studio.close)
            studio.apply_movie((movie, identity(source)))
            studio.analysis = analysis
            studio.voice_pause = 1500
            studio.voice_trim = False
            parts = []
            for i, duration in enumerate((203.25, 202.5, 414.383)):
                path = root / f"voice-{i}.wav"
                filters = ["-af", "volume=enable='between(t,206.5,208)':volume=0"] if i == 2 else []
                ffmpeg("-f", "lavfi", "-i", f"sine=f=1100:r={RATE}:d={duration}", *filters, str(path))
                parts.append(path)
            with patch.object(studio, "background", side_effect=lambda _, work, apply, **kw: apply(work(lambda _: None))), \
                 patch.object(studio, "error", side_effect=lambda message: self.fail(message)), \
                 patch.object(studio, "approve_audio_omissions", side_effect=lambda omissions: self.assertFalse(omissions) or True), \
                 patch("movie_review_studio.app.QMessageBox.information"):
                studio.load_voice_parts(0, parts)
                studio.generate()
                self.assertTrue(studio.voice_sync_valid())
                groups = [c for c in studio.clips if c.movie_audio]
                self.assertEqual([c.role for c in groups], ["Opening", "Highlight", "Highlight", "Highlight", "Ending"])
                self.assertAlmostEqual(studio.voice_recording.duration, 823.133, places=3)
                with wave.open(str(studio.voice_recording.path), "rb") as dry, wave.open(str(studio.voice), "rb") as bed:
                    while original := dry.readframes(RATE * 4):
                        self.assertEqual(bed.readframes(len(original) // 4), original)
                out = root / "verified.mp4"
                with patch("movie_review_studio.app.QFileDialog.getSaveFileName", return_value=(str(out), "")):
                    studio.export()
            info = inspect(out)
            self.assertTrue(info.has_audio)
            self.assertEqual(info.duration, sum(clip.frames for clip in studio.clips))
            report = json.loads(out.with_suffix(".timeline-validation.json").read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            self.assertFalse(report["violations"])
            allocation = json.loads(studio.audio_allocation_path().read_text(encoding="utf-8"))
            self.assertEqual(allocation["dialogue_cues"], 0)
            self.assertEqual(len(allocation["selected"]), 5)
            self.assertTrue(all(row["counts"]["passages_tried"] for row in allocation["sections"]))
            raw = ffmpeg("-i", str(out), "-an", "-vf", "scale=128:72", "-pix_fmt", "rgb24",
                         "-f", "rawvideo", "-threads", "1", "pipe:1", binary=True)
            images = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 72, 128, 3)
            ids = sum((images[:, 4, bit * 11 + 5, 0] > 128).astype(int) << bit for bit in range(11))
            expected = [frame // 4 for row in timeline_entries(studio.clips, movie.fps)
                        for frame in range(row.segment.start, row.segment.end)]
            np.testing.assert_array_equal(ids, expected)
            cursor = 0
            for clip in studio.clips:
                if clip.movie_audio:
                    signal = ffmpeg("-ss", str(cursor + .25), "-i", str(out), "-t", "0.5",
                                    "-ac", "1", "-ar", str(RATE), "-f", "s16le", "pipe:1", binary=True)
                    spectrum = np.abs(np.fft.rfft(np.frombuffer(signal, dtype="<i2")))
                    expected_hz = 330 + 37 * ((clip.start // 4) % 7)
                    self.assertLessEqual(abs(np.argmax(spectrum) * 2 - expected_hz), 2)
                    self.assertLess(spectrum[550], spectrum.max() * .03)
                cursor += clip.frames
            studio.close()
        application.processEvents()
