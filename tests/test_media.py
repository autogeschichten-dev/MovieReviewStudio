from pathlib import Path
from threading import Event, Timer
from time import monotonic
from fractions import Fraction
from dataclasses import replace
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

from strict_fixtures import audio_group

from movie_review_studio.analysis import analyze, fingerprint_highlight, thumbnail_path
from movie_review_studio.media import Cancelled, cancellation, ffmpeg, frame_clock, identity, inspect
from movie_review_studio.model import Clip, timeline_entries, validate
from movie_review_studio.render import Mix, RATE, audio_track, dimensions, render, render_clip, render_selection


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Requires FFmpeg and ffprobe")
class MediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.movie = cls.root / "movie with spaces.mp4"
        inputs = []
        for color in ("red", "green", "blue", "white", "red", "cyan", "blue", "yellow", "red"):
            inputs += ["-f", "lavfi", "-i", f"color={color}:s=96x64:r=12:d=5"]
        ffmpeg(*inputs, "-f", "lavfi", "-i", "sine=f=880:d=45",
               "-filter_complex", "".join(f"[{i}:v]" for i in range(9)) + "concat=n=9:v=1:a=0[v]",
               "-map", "[v]", "-map", "9:a", "-c:v", "libx264", "-c:a", "aac", str(cls.movie))
        cls.media = inspect(cls.movie)

    def test_frame_accurate_cut_at_shot_boundary(self):
        for start, channel in [(59, 0), (60, 1), (119, 1), (120, 2)]:
            with self.subTest(start=start):
                path = self.root / "cut.mp4"
                render_clip(self.media, Clip(start, start + 1, 0, (), (), ()), path, True)
                pixels = ffmpeg("-i", str(path), "-vf", "scale=1:1", "-pix_fmt", "rgb24",
                                "-f", "rawvideo", "-threads", "1", "pipe:1", binary=True)
                self.assertEqual(len(pixels), 3)
                self.assertEqual(np.argmax(list(pixels)), channel)
        self.assertEqual(dimensions(self.media, False), (1620, 1080))

    def test_analysis_decodes_entire_movie_and_uses_cache(self):
        movie = self.root / "pattern.mp4"
        ffmpeg("-f", "lavfi", "-i", "testsrc2=size=160x90:rate=12:duration=3",
               "-f", "lavfi", "-i", "smptebars=size=160x90:rate=12:duration=3",
               "-f", "lavfi", "-i", "color=black:size=160x90:rate=12:duration=2",
               "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0", "-c:v", "libx264", str(movie))
        messages = []
        media = inspect(movie)
        analysis = analyze(media, self.root, messages.append)
        self.assertEqual(analysis.frames, 96)
        self.assertTrue(any(clip.end == 36 for clip in analysis.candidates))
        self.assertTrue(any(clip.start == 36 for clip in analysis.candidates))
        self.assertTrue(all(clip.end <= 72 for clip in analysis.candidates))
        self.assertTrue(all(thumbnail_path(self.root, identity(movie), clip.start).is_file()
                            for clip in analysis.candidates))
        self.assertEqual(analysis, analyze(media, self.root, messages.append))
        self.assertIn("Loaded cached shot analysis.", messages)

    def test_assembled_preview_audio_routes_and_cache(self):
        analysis = analyze(self.media, self.root, lambda _: None)
        highlight = audio_group(Clip(24, 144, 0, (), (), (), highlight=True),
                                [(72, 108), (180, 240), (312, 336)])
        clips = [Clip(0, 12, 0, (), (), ()), highlight, Clip(408, 420, 0, (), (), ())]
        entries = timeline_entries(clips, self.media.fps)
        self.assertEqual([(e.position, e.segment.start, e.segment.end) for e in entries],
                         [(0, 0, 12), (12, 72, 108), (48, 180, 240), (108, 312, 336), (132, 408, 420)])
        self.assertTrue(all(e.segment.frames / self.media.fps <= 5 for e in entries))
        voice = self.root / "voice.wav"
        music = self.root / "music.wav"
        ffmpeg("-f", "lavfi", "-i", "aevalsrc=if(lt(t\\,1)\\,0.1*sin(2*PI*440*t)\\,0.1*sin(2*PI*660*t)):d=2",
               str(voice))
        ffmpeg("-f", "lavfi", "-i", "sine=f=220:d=0.7", str(music))
        mix = self.root / "mix"
        mix.mkdir()
        output = audio_track(self.media, clips, voice, music, mix)
        with wave.open(str(output), "rb") as result:
            self.assertEqual(result.getnframes(), 12 * RATE)
            audio = np.frombuffer(result.readframes(12 * RATE), dtype="<i2")[::2].astype(float)

        def energy(start, frequency):
            data = audio[int(start * RATE):int((start + .1) * RATE)]
            return abs(np.fft.rfft(data)[frequency // 10]) / len(data)

        self.assertGreater(energy(.4, 440), 300)
        self.assertGreater(energy(.4, 220), 80)
        self.assertLess(energy(.4, 880), 3)
        self.assertGreater(energy(5, 880), 300)
        self.assertGreater(energy(3.98, 880), 300)  # Audio continues across an internal video cut at 4 s.
        self.assertLess(abs(audio[RATE + 1]), 100)
        self.assertLess(abs(audio[11 * RATE - 1]), 100)
        self.assertLess(energy(5, 440), 3)
        self.assertLess(energy(5, 220), 3)
        self.assertGreater(energy(11.4, 660), 300)
        self.assertGreater(energy(11.4, 220), 80)
        self.assertGreater(energy(.4, 220), energy(.9, 220) * 2)
        self.assertGreater(energy(11.4, 220), energy(11, 220) * 2)
        path = render(self.media, analysis, clips, voice, music, self.root, lambda _: None)
        self.assertAlmostEqual(inspect(path).duration, 12.0, places=2)
        self.assertEqual(frame_clock(inspect(path))[0], 144)
        self.assertEqual(path, render(self.media, analysis, clips, voice, music, self.root,
                                     lambda _: self.fail("Expected cached preview")))

        def pixels(video):
            raw = ffmpeg("-i", str(video), "-vf", "scale=1:1", "-pix_fmt", "rgb24",
                         "-f", "rawvideo", "-threads", "1", "pipe:1", binary=True)
            return np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)

        assembly_pixels = pixels(path)
        self.assertTrue(np.all(assembly_pixels.max(axis=1) > 80))
        self.assertTrue(np.all(assembly_pixels[12:48, 0] < 8))
        self.assertTrue(np.all(assembly_pixels[48:108].min(axis=1) > 240))
        self.assertTrue(np.all(assembly_pixels[108:132, 0] < 8))
        selected = render_selection(self.media, analysis, clips, 2, voice, music, self.root, lambda _: None)
        np.testing.assert_array_equal(pixels(selected), assembly_pixels[48:108])
        selected_audio = ffmpeg("-i", str(selected), "-map", "0:a:0", "-ac", "1", "-f", "s16le",
                                "pipe:1", binary=True)
        signal = np.frombuffer(selected_audio, dtype="<i2")[:RATE].astype(float)
        self.assertEqual(np.argmax(abs(np.fft.rfft(signal))), 880)

    def test_adaptive_audio_ducks_music_and_keeps_narration_out_of_dialogue(self):
        narration, music = self.root / "duck-voice.wav", self.root / "duck-music.wav"
        ffmpeg("-f", "lavfi", "-i", "sine=f=660:d=1.0", str(narration))
        ffmpeg("-f", "lavfi", "-i", "sine=f=220:d=1.0", str(music))
        clips = [Clip(0, 24, 0, (), (), ()), Clip(30, 54, 0, (), (), (), original=True)]
        folder = self.root / "duck"
        folder.mkdir()
        path = audio_track(self.media, clips, narration, music, folder, Mix())
        with wave.open(str(path), "rb") as stream:
            self.assertEqual(stream.getnframes(), 4 * RATE)
            signal = np.frombuffer(stream.readframes(4 * RATE), dtype="<i2")[::2].astype(float)

        def energy(at, hz):
            samples = signal[round(at * RATE):round((at + .25) * RATE)]
            return abs(np.fft.rfft(samples)[hz // 4])

        self.assertGreater(energy(.25, 660), 1_000_000)
        self.assertGreater(energy(2.5, 880), 1_000_000)
        self.assertLess(energy(2.5, 660), energy(.25, 660) / 50)
        self.assertGreater(energy(1.25, 220), energy(.25, 220) * 1.7)
        self.assertGreater(energy(.25, 220), energy(2.5, 220) * 1.7)

    def test_fractional_fps_cuts(self):
        path = self.root / "fractional.mkv"
        ffmpeg("-f", "lavfi", "-i", "testsrc2=size=96x64:rate=24000/1001:duration=3",
               "-c:v", "libx264", str(path))
        media = inspect(path)
        self.assertEqual(media.fps, Fraction(24000, 1001))
        analysis = analyze(media, self.root, lambda _: None)
        self.assertEqual(analysis.frames, 72)
        render_clip(media, Clip(23, 72, 0, (), (), ()), self.root / "fractional-cut.mp4", True)
        clips = [Clip(29, 72, 0, (), (), ())]
        output = render(media, analysis, clips, None, None, self.root, lambda _: None)
        self.assertEqual(inspect(output).fps, media.fps)
        self.assertEqual(frame_clock(inspect(output))[0], 43)

    def test_cancel_interrupts_a_running_media_command(self):
        stop = Event()
        token = cancellation.set(stop)
        alarm = Timer(.4, stop.set)
        started = monotonic()
        alarm.start()
        try:
            with self.assertRaises(Cancelled):
                ffmpeg("-re", "-f", "lavfi", "-i", "anullsrc", "-t", "10", "-f", "null", "-")
        finally:
            alarm.cancel()
            cancellation.reset(token)
        self.assertLess(monotonic() - started, 5)

    def test_export_keeps_frame_clock_at_mixed_length_joins(self):
        source = self.root / "mixed-length-source.mp4"
        ffmpeg("-f", "lavfi", "-i", "testsrc2=size=96x64:rate=1:duration=26",
               "-c:v", "libx264", str(source))
        media = inspect(source)
        analysis = analyze(media, self.root, lambda _: None)
        clips = [Clip(start, end, 0, (), (), ()) for start, end in ((0, 5), (10, 12), (17, 22))]
        export = self.root / "mixed-length-export.mp4"

        def short_fixture(items, fps, frames, *, final, **policy):
            self.assertTrue(final)
            validate(items, fps, frames, **policy)

        with patch("movie_review_studio.render.validate", side_effect=short_fixture):
            render(media, analysis, clips, None, None, self.root, lambda _: None, destination=export, mix=Mix(), min_gap=1)
        self.assertEqual(frame_clock(inspect(export))[0], 12)

    def test_editor_scene_controls_and_project_round_trip(self):
        from PySide6.QtWidgets import QApplication, QMessageBox
        from movie_review_studio.app import Studio

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        application = QApplication.instance() or QApplication([])
        studio = Studio(self.root)
        self.addCleanup(studio.close)
        source = analyze(self.media, self.root, lambda _: None)
        candidates = [Clip(start, end, 0, (), (), (), shot=start)
                      for start, end in ((0, 12), (80, 128), (200, 248), (320, 344), (430, 442), (520, 532))]
        analysis = replace(source, candidates=candidates)
        highlight = fingerprint_highlight(self.media, analysis, 24, 144, cache=self.root,
                                           picture=[(80, 128), (200, 248), (320, 344)])
        studio.movie, studio.source_key, studio.analysis = self.media, identity(self.movie), analysis
        studio.highlights = [highlight]
        studio.set_clips([candidates[0], highlight, candidates[4]])
        self.assertEqual(studio.source_gap.value(), 5)
        self.assertEqual(studio.validation_report()["adjacent_source_fragment_violations"], 0)
        with self.assertRaisesRegex(ValueError, "configured separation"):
            studio.check([candidates[0], replace(candidates[1], start=30, end=40)])
        studio.source_gap.setValue(0)
        self.assertEqual(studio.source_gap.value(), 5)

        studio.select_row(2)
        self.assertIn("Original movie audio", studio.inspector_details.text())
        studio.select_row(4)
        studio.toggle_clip_audio()
        self.assertTrue(studio.clips[-1].movie_audio)
        studio.undo()
        self.assertFalse(studio.clips[-1].movie_audio)
        studio.redo()
        self.assertTrue(studio.clips[-1].movie_audio)
        studio.undo()

        audio = self.root / "project-audio.wav"
        ffmpeg("-f", "lavfi", "-i", "sine=f=220:d=1", str(audio))
        studio.voice = studio.music = audio
        studio.target.setValue(1080)
        project = self.root / "test-project.mrs.json"
        with patch("movie_review_studio.app.QFileDialog.getSaveFileName", return_value=(str(project), "")):
            studio.save_project()
        expected = studio.clips[:]
        self.assertEqual(json.loads(project.read_text(encoding="utf-8"))["highlights"][0][6],
                         [[80, 128], [200, 248], [320, 344]])

        def run_job(_message, operation, complete):
            complete(operation(lambda _: None))

        with patch("movie_review_studio.app.analyze", return_value=analysis), \
             patch("movie_review_studio.app.QFileDialog.getOpenFileName", return_value=(str(project), "")), \
             patch("movie_review_studio.app.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes), \
             patch.object(studio, "background", side_effect=run_job):
            studio.set_clips([])
            studio.open_project()
            self.assertEqual(studio.clips, expected)
            self.assertEqual((studio.voice, studio.music), (audio, audio))
            legacy = json.loads(project.read_text(encoding="utf-8"))
            legacy.update(version=1, target=18, highlights=[[24, 144]], minimum_source_gap_seconds=0,
                          clips=[item[:3] for item in legacy["clips"]])
            project.write_text(json.dumps(legacy), encoding="utf-8")
            studio.open_project()
            self.assertEqual(studio.source_gap.value(), 5)
            self.assertGreater(studio.validation_report()["touching_source_interval_violations"], 0)
            with patch("movie_review_studio.app.QFileDialog.getSaveFileName") as picker:
                with self.assertRaisesRegex(ValueError, "touch|configured separation"):
                    studio.export()
                picker.assert_not_called()
        application.processEvents()
