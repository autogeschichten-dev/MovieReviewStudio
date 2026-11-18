from pathlib import Path
from fractions import Fraction
import shutil
import tempfile
import unittest
import wave

import numpy as np

from movie_review_studio.analysis import analyze, fingerprint_highlight
from movie_review_studio.media import ffmpeg, inspect
from movie_review_studio.model import Clip
from movie_review_studio.render import RATE, audio_track, dimensions, render, render_clip


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Requires FFmpeg and ffprobe")
class MediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.movie = cls.root / "movie with spaces.mp4"
        inputs = []
        for color in ("red", "green", "blue", "white"):
            inputs += ["-f", "lavfi", "-i", f"color={color}:s=96x64:r=12:d=5"]
        ffmpeg(*inputs, "-f", "lavfi", "-i", "sine=f=880:d=20",
               "-filter_complex", "[0:v][1:v][2:v][3:v]concat=n=4:v=1:a=0[v]",
               "-map", "[v]", "-map", "4:a", "-c:v", "libx264", "-c:a", "aac", str(cls.movie))
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
        self.assertEqual(analysis, analyze(media, self.root, messages.append))
        self.assertIn("Loaded cached shot analysis.", messages)

    def test_assembled_preview_audio_routes_and_cache(self):
        analysis = analyze(self.media, self.root, lambda _: None)
        highlight = fingerprint_highlight(self.media, analysis, 24, 144)
        clips = [Clip(0, 12, 0, (), (), ()), highlight, Clip(150, 162, 0, (), (), ())]
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
        self.assertLess(energy(5, 440), 3)
        self.assertLess(energy(5, 220), 3)
        self.assertGreater(energy(11.4, 660), 300)
        self.assertGreater(energy(11.4, 220), 80)
        self.assertGreater(energy(.4, 220), energy(.9, 220) * 2)
        self.assertGreater(energy(11.4, 220), energy(11, 220) * 2)
        path = render(self.media, analysis, clips, voice, music, self.root, lambda _: None)
        self.assertAlmostEqual(inspect(path).duration, 12.0, places=2)
        self.assertEqual(path, render(self.media, analysis, clips, voice, music, self.root,
                                     lambda _: self.fail("Expected cached preview")))

    def test_fractional_fps_cuts(self):
        path = self.root / "fractional.mkv"
        ffmpeg("-f", "lavfi", "-i", "testsrc2=size=96x64:rate=24000/1001:duration=3",
               "-c:v", "libx264", str(path))
        media = inspect(path)
        self.assertEqual(media.fps, Fraction(24000, 1001))
        analysis = analyze(media, self.root, lambda _: None)
        self.assertEqual(analysis.frames, 72)
        render_clip(media, Clip(23, 72, 0, (), (), ()), self.root / "fractional-cut.mp4", True)
