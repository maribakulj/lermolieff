"""The head: a one-hidden-layer network over a frozen encoder's vector, trained with an asymmetric loss on codes and their ancestors."""

from __future__ import annotations

import numpy as np
import torch

from .hierarchy import ancestors, most_specific

MIN_COUNT = 5  # codes seen on at least 5 training images, with their ancestors


def build_vocab(labels: list[list[str]]) -> list[str]:
    counts: dict[str, int] = {}
    for y in labels:
        for c in set(y):
            counts[c] = counts.get(c, 0) + 1
    return sorted({a for c, n in counts.items() if n >= MIN_COUNT for a in ancestors(c)})


def targets(labels: list[list[str]], vocab: list[str]) -> np.ndarray:
    vi = {c: i for i, c in enumerate(vocab)}
    t = np.zeros((len(labels), len(vocab)), dtype=np.float32)
    for r, y in enumerate(labels):
        for c in y:
            for a in ancestors(c):
                if a in vi:
                    t[r, vi[a]] = 1
    return t


def make_head(in_dim: int, n_codes: int) -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(in_dim, 2048), torch.nn.GELU(), torch.nn.Dropout(0.2),
                               torch.nn.Linear(2048, n_codes))


def asym_loss(logits, y, gn=4.0, gp=0.0, clip=0.05):
    """Asymmetric loss; every absent label is a negative (catalogue incompleteness is not masked)."""
    p = torch.sigmoid(logits)
    pn = (p - clip).clamp(min=0)
    lp = y * (1 - p) ** gp * torch.log(p.clamp(min=1e-8))
    ln = (1 - y) * pn ** gn * torch.log((1 - pn).clamp(min=1e-8))
    return -(lp + ln).sum(1).mean()


class Head:
    """A trained head (head.pt): its vocabulary, threshold and the encoder configuration it was trained on."""

    def __init__(self, path, device: str = "cpu"):
        ck = torch.load(path, map_location=device, weights_only=True)
        self.vocab, self.threshold, self.feature_config = ck["vocab"], float(ck["threshold"]), ck["feature_config"]
        in_dim = ck["state"]["0.weight"].shape[1]
        self.net = make_head(in_dim, len(self.vocab)).to(device).eval()
        self.net.load_state_dict(ck["state"])
        self.device = device

    @torch.no_grad()
    def probs(self, x: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        return torch.cat([torch.sigmoid(self.net(x[i:i + 4096].to(self.device))) for i in range(0, len(x), 4096)]).cpu().numpy()

    def predict(self, x: np.ndarray, threshold: float | None = None) -> list[list[str]]:
        t = self.threshold if threshold is None else threshold
        return [most_specific([self.vocab[j] for j in np.nonzero(row >= t)[0]]) for row in self.probs(x)]
