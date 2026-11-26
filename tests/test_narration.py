from fractions import Fraction
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import numpy as np

from movie_review_studio import director, narration
from movie_review_studio.media import ffmpeg, identity
from movie_review_studio.model import Clip, source_gap
from movie_review_studio.render import RATE, audio_track


def clip(start, frames=5, *, original=False):
    return Clip(start, start + frames, 1, (), (), (), shot=start, original=original)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools required")
class NarrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.progress = lambda _: None

    def tone(self, name, volume, frequency=440, pad=True):
        path = self.root / name
        filters = f"volume={volume}"
        if pad:
            filters += ",adelay=500,apad=pad_dur=0.5"
        ffmpeg("-f", "lavfi", "-i", f"sine=f={frequency}:r=22050:d=1", "-af", filters, str(path))
        return narration.read_part(path, self.root, self.progress)

    def merged(self, parts, delay=250, trim=True):
        return narration.merge(parts, delay, trim, self.root, self.progress)

    def samples(self, path):
        with wave.open(str(path), "rb") as audio:
            self.assertEqual((audio.getnchannels(), audio.getframerate(), audio.getsampwidth()), (2, RATE, 2))
            return np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").reshape(-1, 2)

    def test_single_wav_and_mp3_merge_trim_pause_and_loudness(self):
        first = self.tone("a.wav", 1)
        second = self.tone("b.mp3", 4, frequency=660)
        self.assertTrue(first.peaks and second.peaks)
        single = self.merged([first], trim=False)
        self.assertEqual(single.frames, 2 * RATE)
        merged = self.merged([first, second])
        (a, a_head, a_length), (b, b_head, b_length) = merged.boundaries
        self.assertEqual(a, 0)
        self.assertGreaterEqual(a_head, round(.33 * RATE))
        self.assertGreaterEqual(b_head, round(.30 * RATE))
        self.assertAlmostEqual(a_length / RATE, 1.30, delta=.03)
        self.assertAlmostEqual(b_length / RATE, 1.30, delta=.04)
        self.assertEqual(b - a_length, RATE // 4)
        samples = self.samples(merged.path)
        self.assertEqual(len(samples), merged.frames)
        self.assertFalse(np.any(samples[a_length:b]))
        first_active = samples[RATE // 3:RATE * 3 // 4, 0].astype(float)
        second_active = samples[b + RATE // 3:b + RATE * 3 // 4, 0].astype(float)
        ratio = np.sqrt(np.mean(first_active ** 2) / np.mean(second_active ** 2))
        self.assertGreater(ratio, .75)
        self.assertLess(ratio, 1.25)
        crossings = np.count_nonzero(np.diff(first_active > 0)) / 2
        self.assertAlmostEqual(crossings / (len(first_active) / RATE), 440, delta=3)
        with patch.object(narration, "pcm", side_effect=AssertionError("cache should reuse the merge")):
            self.assertEqual(self.merged([first, second]), merged)
        reverse = self.merged([second, first], delay=0)
        self.assertEqual(reverse.boundaries[1][0], b_length)
        self.assertFalse(np.array_equal(self.samples(reverse.path)[:RATE], samples[:RATE]))

    def test_non_edge_pause_and_very_short_input_are_retained(self):
        samples = np.full((3 * RATE, 2), 1500, dtype="<i2")
        samples[RATE:2 * RATE] = 0
        self.assertEqual(narration.trim_edges(samples), (0, 3 * RATE))
        samples = np.zeros((100, 2), dtype="<i2")
        samples[-1, 1] = 1500
        self.assertEqual(narration.trim_edges(samples), (0, 100))

    def test_synchronized_bed_preserves_every_sample_and_original_audio(self):
        parts = [self.tone("a.wav", 1), self.tone("b.wav", 2)]
        merged = self.merged(parts, delay=500, trim=False)
        movie = SimpleNamespace(fps=Fraction(2), has_audio=True)
        analysis = SimpleNamespace(frames=100)
        # Two seconds of picture, a dialogue cut, then additional picture.
        clips = [clip(0, 4), clip(20, 2, original=True), clip(40, 6), clip(60, 4)]
        edits, _, schedule, length, aligned = narration.synchronize(
            merged, clips, [], [], movie, analysis, 5, True, self.progress)
        self.assertFalse(aligned)
        self.assertEqual(edits, clips[:3])
        self.assertTrue(edits[1].movie_audio)
        self.assertTrue(all(source_gap(a, b) >= 5 * movie.fps for a, b in zip(edits, edits[1:])))
        bed = self.samples(narration.materialize(merged, schedule, length, self.root))
        recording = self.samples(merged.path)
        for at, start, end in schedule:
            np.testing.assert_array_equal(bed[at:at + end - start], recording[start:end])
        self.assertEqual(narration.markers(merged, schedule), ((1, 0.0), (2, 2.5)))
        self.assertEqual(sum(end - start for _, start, end in schedule), merged.frames)
        self.assertTrue(all(item.frames / movie.fps <= 5 for item in edits))
        original = self.root / "movie-audio.wav"
        ffmpeg("-f", "lavfi", "-i", "sine=f=150:r=48000:d=25", str(original))
        movie.path, movie.offset = original, 0
        mixed = self.samples(audio_track(movie, edits, narration.materialize(merged, schedule, length, self.root),
                                         None, self.root, managed_voice=True))
        self.assertEqual(len(mixed), 6 * RATE)
        self.assertTrue(np.any(mixed[2 * RATE:3 * RATE]))
        np.testing.assert_allclose(mixed[:2 * RATE], bed[:2 * RATE] * .92, atol=2)
        np.testing.assert_allclose(mixed[3 * RATE:], bed[2 * RATE:] * .92, atol=2)
        with self.assertRaisesRegex(ValueError, "does not match this edit"):
            audio_track(movie, clips, narration.materialize(merged, schedule, length, self.root),
                        None, self.root, managed_voice=True)
        with self.assertRaisesRegex(ValueError, "configured separation"):
            narration.synchronize(merged, [clips[0], clip(7, 2), *clips[1:]], [], [], movie, analysis, 5, False, self.progress)

    def test_refuses_to_cut_voice_when_a_highlight_has_no_safe_pause(self):
        path = self.root / "continuous.wav"
        ffmpeg("-f", "lavfi", "-i", "sine=f=280:d=3", str(path))
        merged = self.merged([narration.read_part(path, self.root, self.progress)])
        movie = SimpleNamespace(fps=Fraction(1), has_audio=True)
        with self.assertRaisesRegex(ValueError, "verified speech pause"):
            narration.synchronize(merged, [clip(0, 2), clip(10, 2, original=True), clip(20, 2)],
                                  [], [], movie, SimpleNamespace(frames=40), 5, False, self.progress)

    def test_timestamped_sentences_anchor_to_relevant_footage(self):
        parts = [self.tone("a.wav", 1), self.tone("b.wav", 1)]
        merged = self.merged(parts, delay=500, trim=False)
        script = [director.ScriptSection("part-1", "The family works. Wooden shutters close.",
                                        ((0, 3, "The family works"), (10, 13, "Wooden shutters close")))]
        for part, line in zip(parts, ("The family works.", "Wooden shutters close.")):
            part.path.with_suffix(".srt").write_text(
                f"1\n00:00:00,500 --> 00:00:01,500\n{line}\n", encoding="utf-8")
        cues = narration.transcript(merged, parts, None, self.root, self.progress)
        self.assertEqual([c["start"] for c in cues], [.5, 3])
        edits, updated, schedule, _, aligned = narration.synchronize(
            merged, [clip(0, 3), clip(10, 3)], script, cues, SimpleNamespace(fps=Fraction(1), has_audio=False),
            SimpleNamespace(frames=30), 5, False, self.progress)
        self.assertTrue(aligned)
        self.assertEqual(len(edits), 2)
        self.assertEqual(updated[0].duration, merged.duration)
        self.assertEqual(schedule[1][0], 3 * RATE)
        self.assertGreater(schedule[1][1], 2 * RATE)

    def test_manager_add_replace_reorder_save_and_restore_with_missing_derived_audio(self):
        from PySide6.QtWidgets import QApplication, QMessageBox
        from movie_review_studio.app import Studio
        from movie_review_studio.media import inspect

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        application = QApplication.instance() or QApplication([])
        studio = Studio(self.root)
        studio.duration_mode = "fixed"
        self.addCleanup(studio.close)
        studio.open_voice_manager()
        self.assertEqual(len(studio.voice_parts), 3)
        a, b, c, d = (self.tone(name, 1, pad=False) for name in ("a.wav", "b.mp3", "c.wav", "d.wav"))
        movie = self.root / "movie.mp4"
        ffmpeg("-f", "lavfi", "-i", "testsrc2=s=96x64:r=2:d=20", "-c:v", "libx264", str(movie))
        picture = [clip(0, 8), clip(18, 8)]
        analysis = SimpleNamespace(candidates=picture, frames=40, offset=0)
        studio.movie, studio.source_key, studio.analysis = inspect(movie), identity(movie), analysis

        def inline(_message, operation, apply):
            apply(operation(self.progress))

        with patch.object(studio, "background", side_effect=inline):
            studio.load_voice_parts(0, [a.path])
            self.assertEqual(studio.voice_parts, (a, None, None))
            studio.load_voice_parts(1, [b.path, c.path])
            studio.voice_manager.move(2, -1)
            studio.load_voice_parts(2, [d.path])
            self.assertEqual(studio.voice_parts, (a, c, d))
            studio.set_clips(picture)
            studio.voice_options(250, True, True, None)
            studio.process_voice("sync")
            self.assertTrue(studio.voice_sync_valid())
            self.assertEqual(studio.clips, picture[:1])
            exported = self.root / "combined.mp3"
            studio.process_voice("export", destination=exported)
            self.assertAlmostEqual(inspect(exported, video=False).duration, studio.voice_recording.duration, delta=.05)
            studio.set_clips(picture)
            self.assertFalse(studio.voice_sync_valid())
            studio.undo()
            self.assertTrue(studio.voice_sync_valid())
            self.assertTrue(studio.validation_report()["passed"])
            project = self.root / "review.mrs.json"
            with patch("movie_review_studio.app.QFileDialog.getSaveFileName", return_value=(str(project), "")):
                studio.save_project()
            studio.voice.unlink()
            with patch("movie_review_studio.app.QFileDialog.getOpenFileName", return_value=(str(project), "")), \
                 patch("movie_review_studio.app.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes), \
                 patch("movie_review_studio.app.analyze", return_value=analysis):
                studio.open_project()
            self.assertTrue(studio.voice.is_file() and studio.voice_sync_valid())
            self.assertEqual(studio.voice_parts, (a, c, d))
            studio.voice_manager.remove(1)
            self.assertEqual(studio.voice_parts, (a, d, None))
            self.assertIsNone(studio.voice_sync)
            studio.undo()
            self.assertTrue(studio.voice_sync_valid())
