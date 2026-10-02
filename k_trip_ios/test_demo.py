"""Run: python -m unittest test_demo -v (with local .packages on PYTHONPATH)."""
import base64
from io import BytesIO
import unittest
import numpy as np
from PIL import Image
from torchvision import transforms
from fastapi.testclient import TestClient
from server import app
from src.pipeline import preprocess, load_meta
from pathlib import Path

class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()
        rng = np.random.default_rng(42)
        cls.image = rng.integers(0, 256, (479, 641, 3), dtype=np.uint8)
        buf = BytesIO()
        Image.fromarray(cls.image).save(buf, format="JPEG")
        cls.data = buf.getvalue()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_preprocessing_matches_training_on_odd_sizes(self):
        meta = load_meta(Path(__file__).parent / "checkpoints" / "meta.json")
        transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(meta["mean"],meta["std"])])
        expected = transform(Image.fromarray(self.image)).numpy()
        actual = preprocess(self.image[:, :, ::-1], meta)
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_capture_and_live_after_gradcam(self):
        response = self.client.post("/api/capture", content=self.data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_arch"], "convnext_tiny")
        self.assertEqual(response.json()["resize"], "stretch")
        result = response.json()
        self.assertAlmostEqual(sum(result["probs"].values()),1, places=5)
        overlay = Image.open(BytesIO(base64.b64decode(result["overlay"].split(",")[1])))
        self.assertEqual(overlay.size, (224,224))
        with self.client.websocket_connect("/api/live") as socket:
            for _ in range(3):
                socket.send_bytes(self.data)
                live = socket.receive_json()
                self.assertEqual(live["label"],result["label"])
                for k in live["probs"]:
                    self.assertAlmostEqual(live["probs"][k],result["probs"][k],places=5)
            socket.send_bytes(b"invalid")
            self.assertIn("error",socket.receive_json())
            socket.send_bytes(self.data)
            self.assertIn("probs",socket.receive_json())

    def test_bad_and_oversized_images(self):
        self.assertEqual(self.client.post("/api/capture",content=b"bad").status_code,400)
        self.assertEqual(self.client.post("/api/capture",content=b"x"*(6*1024*1024+1)).status_code,413)

    def test_routes(self):
        self.assertEqual(self.client.get("/").status_code,200)
        self.assertEqual(self.client.get("/static/app.js").status_code,200)
        self.assertEqual(self.client.get("/api/health").json()["model"]["classes"],["mug","straight","taper_smooth","taper_step"])

if __name__ == "__main__":
    unittest.main()
