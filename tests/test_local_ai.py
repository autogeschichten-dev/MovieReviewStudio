from base64 import b64decode
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import json
import shutil

from movie_review_studio import local_ai
from movie_review_studio.media import ffmpeg
from movie_review_studio.recap_upstream import frames


class LocalAICases(TestCase):
    def test_real_extracted_jpeg_through_loopback_mock(self):
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            self.skipTest("FFmpeg required")
        calls = []

        class OllamaFixture(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def reply(self, value):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(value).encode())

            def do_GET(self):
                self.reply({"version": "test"} if self.path == "/api/version" else
                           {"models": [{"name": "moondream:latest", "digest": "fixture"}]})

            def do_POST(self):
                calls.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                self.reply({"response": json.dumps({"objects": ["red rectangle"], "uncertain": ["fixture"]})})

        with TemporaryDirectory() as folder, HTTPServer(("127.0.0.1", 0), OllamaFixture) as server:
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                root = Path(folder)
                movie = root / "fixture.mp4"
                image = root / "frame.jpg"
                ffmpeg("-f", "lavfi", "-i", "testsrc2=s=128x96:r=4:d=3", "-c:v", "libx264", str(movie))
                frames.extract_frames(movie, [1.25], image, width=128, labels=False)
                with patch.object(local_ai, "OLLAMA", f"http://127.0.0.1:{server.server_port}"):
                    self.assertEqual(local_ai.discover(), ("test", {"moondream:latest": "fixture"}))
                    trace = []
                    result = local_ai.describe_image(image, "moondream:latest", 1.25, trace=trace.append)
                    self.assertEqual(trace[-1]["http_status"], 200)
                    self.assertIn("red rectangle", trace[-1]["body"])
                    self.assertEqual(result["timestamp"], 1.25)
                    batch = local_ai.VisionBatch(SimpleNamespace(path=movie, fps=4, duration=3),
                                                SimpleNamespace(offset=0), root, "moondream:latest", "fixture")
                    clip = SimpleNamespace(start=0, end=8, frames=8, production_card=False)
                    first = batch.describe([clip], lambda _: None)
                    self.assertEqual(batch.describe([clip], lambda _: None), first)
                route, body = calls[0]
                self.assertEqual(route, "/api/generate")
                self.assertNotIn("format", body)
                self.assertEqual(b64decode(body["images"][0], validate=True), image.read_bytes())
                self.assertEqual(result["uncertain"], ["fixture"])
                self.assertEqual(len(calls), 2)
            finally:
                server.shutdown()
                thread.join()

    def test_plain_caption_and_uncertainty(self):
        result = local_ai.normalize_observation("Two people stand by a door. It might be a school.")
        self.assertEqual(result["description"], "Two people stand by a door.")
        self.assertEqual(result["uncertain"], ["It might be a school."])
        self.assertEqual(local_ai.normalize_observation('{"description":"A red square."}')["description"],
                         "A red square.")

    def test_malformed_json_and_empty_descriptions_are_retryable(self):
        for bad in ('{"description":', "  ", '{"uncertain":["Something might be there"]}', "None"):
            with self.subTest(bad=bad), self.assertRaises(local_ai.ObservationError):
                local_ai.normalize_observation(bad)

    def test_timeout_reports_transport_and_invalid_jpeg_is_never_sent(self):
        with patch.object(local_ai, "build_opener") as opener:
            opener.return_value.open.side_effect = TimeoutError("fixture timeout")
            trace = []
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                local_ai.request("/api/generate", {}, trace=trace.append)
            self.assertIn("transport_error", trace[-1])
        with TemporaryDirectory() as folder, patch.object(local_ai, "request") as send:
            image = Path(folder) / "corrupt.jpg"
            image.write_bytes(b"\xff\xd8corrupt")
            with self.assertRaises(ValueError):
                local_ai.describe_image(image, "moondream", 0)
            send.assert_not_called()

    def test_partial_failures_retry_and_keep_later_observations(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.mp4"
            source.write_bytes(b"fixture")
            movie = SimpleNamespace(path=source, fps=10, duration=3)
            analysis = SimpleNamespace(offset=0)
            batch = local_ai.VisionBatch(movie, analysis, root, "moondream", "fixture")
            clips = [SimpleNamespace(start=i, end=i + 10, frames=10, production_card=False) for i in (0, 10, 20)]
            report = []
            def extract(_movie, _times, path, **_options):
                path.write_bytes(b"fixture")
            with patch.object(local_ai.frames, "extract_frames", side_effect=extract), \
                 patch.object(local_ai, "usable_frame", return_value=True), \
                 patch.object(local_ai, "describe_image", side_effect=[
                     local_ai.ObservationError("empty"), local_ai.ObservationError("empty"), {"description": "Door"},
                     local_ai.ObservationError("empty"), {"description": "Window"}]) as vision:
                found = batch.describe(clips, lambda _: None, observe=report.append)
                self.assertEqual(list(found), [10, 20])
                self.assertEqual(batch.failed, 1)
                self.assertEqual(report[-1]["successful"], 2)
                self.assertEqual(vision.call_args.kwargs["prompt"], local_ai.RETRY_PROMPT)
            with patch.object(local_ai, "describe_image", side_effect=RuntimeError("still offline")), \
                 patch.object(local_ai.frames, "extract_frames", side_effect=extract), \
                 patch.object(local_ai, "usable_frame", return_value=True):
                self.assertEqual(list(batch.describe(clips, lambda _: None)), [10, 20])

    def test_advertised_moondream_cannot_be_text_model(self):
        from movie_review_studio.director import VisionProvider
        with self.assertRaisesRegex(ValueError, "separate text model"):
            VisionProvider.from_environment("moondream:latest", text_only=True)
