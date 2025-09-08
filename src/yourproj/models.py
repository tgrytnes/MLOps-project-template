from __future__ import annotations
from typing import Any
import numpy as np

from .device import pick_compute_env


class _SKLearnWrapper:
    def __init__(self):
        from sklearn.linear_model import LogisticRegression

        self.model = LogisticRegression(max_iter=200)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_SKLearnWrapper":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class _TorchWrapper:
    def __init__(self, n_features: int, device: str):
        import torch
        import torch.nn as nn

        self.torch = torch
        self.device = torch.device(device)
        self.model = nn.Sequential(nn.Linear(n_features, 1)).to(self.device)
        self.loss = nn.BCEWithLogitsLoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-2)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 256) -> "_TorchWrapper":
        torch = self.torch
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32, device=self.device)

        n = X_t.shape[0]
        for _ in range(epochs):
            for i in range(0, n, batch_size):
                xb = X_t[i : i + batch_size]
                yb = y_t[i : i + batch_size]
                self.opt.zero_grad()
                logits = self.model(xb)
                loss = self.loss(logits, yb)
                loss.backward()
                self.opt.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        torch = self.torch
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            logits = self.model(X_t)
            probs_pos = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        probs_neg = 1.0 - probs_pos
        return np.stack([probs_neg, probs_pos], axis=1)


class _TFWrapper:
    def __init__(self, n_features: int):
        import tensorflow as tf

        self.tf = tf
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(1),  # logits
        ])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-2),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        )

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 256) -> "_TFWrapper":
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import numpy as np

        logits = self.model.predict(X, verbose=0).reshape(-1)
        probs_pos = 1.0 / (1.0 + np.exp(-logits))
        probs_neg = 1.0 - probs_pos
        return np.stack([probs_neg, probs_pos], axis=1)


def train_logreg(X: Any, y: Any, backend: str | None = None):
    """
    Train a simple logistic model with automatic backend/device selection.

    - If PyTorch is installed and a GPU is available (CUDA/ROCm/MPS), trains with PyTorch on that device.
    - Else if TensorFlow is installed and a GPU is available, trains with TF.
    - Else falls back to scikit-learn on CPU.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    env = pick_compute_env(backend or "auto")

    if env.framework == "torch":
        model = _TorchWrapper(n_features=X.shape[1], device=env.device)
        return model.fit(X, y)
    if env.framework == "tensorflow":
        model = _TFWrapper(n_features=X.shape[1])
        return model.fit(X, y)

    # sklearn fallback
    model = _SKLearnWrapper()
    return model.fit(X, y)
