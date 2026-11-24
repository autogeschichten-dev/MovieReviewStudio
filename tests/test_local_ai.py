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
                ffmpeg("-f", "lavfi", "-i", "color=red:s=128x96:r=4:d=3", "-c:v", "libx264", str(movie))
                frames.extract_frames(movie, [1.25], image, width=128, labels=False)
                with patch.object(local_ai, "OLLAMA", f"http://127.0.0.1:{server.server_port}"):
                    self.assertEqual(local_ai.discover(), ("test", {"moondream:latest": "fixture"}))
                    result = local_ai.describe_image(image, "moondream:latest", 1.25)
                    self.assertEqual(result["timestamp"], 1.25)
                    batch = local_ai.VisionBatch(SimpleNamespace(path=movie, fps=4),
                                                SimpleNamespace(offset=0), root, "moondream:latest", "fixture")
                    clip = SimpleNamespace(start=0, end=8, frames=8)
                    first = batch.describe([clip], lambda _: None)
                    self.assertEqual(batch.describe([clip], lambda _: None), first)
                route, body = calls[0]
                self.assertEqual(route, "/api/generate")
                self.assertEqual(b64decode(body["images"][0], validate=True), image.read_bytes())
                self.assertEqual(result["uncertain"], ["fixture"])
                self.assertEqual(len(calls), 2)
            finally:
                server.shutdown()
                thread.join()

    def test_advertised_moondream_cannot_be_text_model(self):
        from movie_review_studio.director import VisionProvider
        with self.assertRaisesRegex(ValueError, "separate text model"):
            VisionProvider.from_environment("moondream:latest", text_only=True)
