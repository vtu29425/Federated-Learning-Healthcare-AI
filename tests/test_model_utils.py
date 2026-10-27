import unittest
import numpy as np
from model_utils import sigmoid, accuracy, fedavg, clip_and_add_noise


class ModelUtilsTests(unittest.TestCase):
    def test_sigmoid_midpoint(self):
        self.assertAlmostEqual(float(sigmoid(np.array([0.0]))[0]), 0.5)

    def test_accuracy(self):
        y = np.array([0.0, 1.0, 1.0])
        p = np.array([0.1, 0.9, 0.2])
        self.assertAlmostEqual(accuracy(y, p), 2 / 3)

    def test_fedavg_weighted(self):
        w, b = fedavg([
            (np.array([1.0, 3.0]), 2.0, 1),
            (np.array([3.0, 5.0]), 4.0, 3),
        ])
        np.testing.assert_allclose(w, np.array([2.5, 4.5]))
        self.assertAlmostEqual(b, 3.5)

    def test_clip(self):
        w, b = clip_and_add_noise(
            np.array([10.0, 0.0]), 0.0, 1.0, 0.0, np.random.default_rng(1)
        )
        self.assertLessEqual(np.linalg.norm(np.array([*w, b])), 1.000001)


if __name__ == "__main__":
    unittest.main()
