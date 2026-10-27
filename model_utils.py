"""Small NumPy utilities for the educational federated-learning demo."""
from __future__ import annotations
import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def predict_proba(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return sigmoid(X @ weights + bias)


def binary_cross_entropy(y: np.ndarray, probabilities: np.ndarray) -> float:
    eps = 1e-9
    p = np.clip(probabilities, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def accuracy(y: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((probabilities >= 0.5).astype(int) == y.astype(int)))


def fedavg(client_parameters: list[tuple[np.ndarray, float, int]]) -> tuple[np.ndarray, float]:
    """Weighted average of (weights, bias, number_of_examples) from clients."""
    total = sum(n for _, _, n in client_parameters)
    if total <= 0:
        raise ValueError("Total client sample count must be positive.")
    w = sum(weights * n for weights, _, n in client_parameters) / total
    b = sum(bias * n for _, bias, n in client_parameters) / total
    return w, float(b)


def clip_and_add_noise(
    weights: np.ndarray,
    bias: float,
    clip_norm: float,
    noise_std: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Illustrative update clipping/noise; not a complete DP-SGD implementation."""
    vector = np.concatenate([weights, np.array([bias])])
    norm = float(np.linalg.norm(vector))
    if clip_norm > 0 and norm > clip_norm:
        vector = vector * (clip_norm / norm)
    if noise_std > 0:
        vector = vector + rng.normal(0.0, noise_std, size=vector.shape)
    return vector[:-1], float(vector[-1])
