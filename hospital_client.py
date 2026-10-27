"""Simulated hospital client: data stays local and only parameters are returned."""
from __future__ import annotations
import numpy as np
from model_utils import predict_proba, binary_cross_entropy, clip_and_add_noise


FEATURE_NAMES = [
    "synthetic_feature_1",
    "synthetic_feature_2",
    "synthetic_feature_3",
    "synthetic_feature_4",
    "synthetic_feature_5",
    "synthetic_feature_6",
]


def make_synthetic_hospital_data(
    n_samples: int, seed: int, hospital_shift: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Create fictional numeric data; labels are synthetic and medically meaningless."""
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=hospital_shift, scale=1.0, size=(n_samples, len(FEATURE_NAMES)))
    true_w = np.array([0.8, -0.6, 0.45, 0.2, -0.35, 0.55])
    logits = X @ true_w + rng.normal(0, 0.65, size=n_samples)
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
    y = (rng.random(n_samples) < probabilities).astype(float)
    return X, y


class HospitalClient:
    def __init__(self, name: str, X: np.ndarray, y: np.ndarray, seed: int):
        self.name = name
        self.X = X
        self.y = y
        self.rng = np.random.default_rng(seed)

    def train(
        self,
        global_weights: np.ndarray,
        global_bias: float,
        local_epochs: int = 2,
        learning_rate: float = 0.08,
        dp_clip_norm: float = 5.0,
        dp_noise: float = 0.0,
    ) -> tuple[np.ndarray, float, int, float]:
        weights = global_weights.copy()
        bias = float(global_bias)
        n = len(self.y)
        for _ in range(local_epochs):
            p = predict_proba(self.X, weights, bias)
            grad_w = self.X.T @ (p - self.y) / n
            grad_b = float(np.mean(p - self.y))
            weights -= learning_rate * grad_w
            bias -= learning_rate * grad_b

        local_loss = binary_cross_entropy(
            self.y, predict_proba(self.X, weights, bias)
        )
        weights, bias = clip_and_add_noise(
            weights, bias, dp_clip_norm, dp_noise, self.rng
        )
        return weights, bias, n, local_loss
