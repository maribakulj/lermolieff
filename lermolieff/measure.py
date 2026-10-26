"""Measures: hierarchical F1 by the notation, exact codes, per-branch F1, paired bootstrap (by images or by groups such as books)."""

from __future__ import annotations

import numpy as np

from .hierarchy import ancestors, terms


def hier_f1(gt: list[str], pred: list[str]) -> float:
    """F1 between the ancestor sets of the truth and of the prediction (both closed upward); 0 if either is empty."""
    if not pred or not gt:
        return 0.0
    G, P = set().union(*map(ancestors, gt)), set().union(*map(ancestors, pred))
    i = len(G & P)
    return 0.0 if i == 0 else 2 * i / (len(G) + len(P))


def per_image(gts: list[list[str]], preds: list[list[str]]) -> np.ndarray:
    return np.array([hier_f1(g, p) for g, p in zip(gts, preds, strict=True)])


def exact(gts: list[list[str]], preds: list[list[str]]) -> dict:
    """Micro precision, recall and F1 on the exact code strings."""
    tp = sum(len(set(g) & set(p)) for g, p in zip(gts, preds, strict=True))
    npred, ngt = sum(len(set(p)) for p in preds), sum(len(set(g)) for g in gts)
    pr, rc = tp / max(npred, 1), tp / max(ngt, 1)
    return {"precision": pr, "recall": rc, "f1": 0 if pr + rc == 0 else 2 * pr * rc / (pr + rc)}


def branch_terms(codes: list[str], b: str) -> list[str]:
    """Strict projection on a branch: the terms of b, compounds cut into their parts."""
    return [t for c in codes for t in terms(c) if t.startswith(b)]


def branch_f1(gt: list[str], pred: list[str], b: str) -> float | None:
    """Hierarchical F1 restricted to branch b (None when the truth has no code of b)."""
    g = branch_terms(gt, b)
    if not g:
        return None
    p = branch_terms(pred, b)
    if not p:
        return 0.0
    G = {a for t in g for a in ancestors(t) if a.startswith(b)}
    P = {a for t in p for a in ancestors(t) if a.startswith(b)}
    i = len(G & P)
    return 0.0 if i == 0 else 2 * i / (len(G) + len(P))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, groups: list | None = None, draws: int = 2000, seed: int = 0,
                     level: float = 0.95) -> dict:
    """Difference of means a - b with a percentile interval; groups (e.g. books) are drawn whole when given."""
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = [rng.integers(0, len(a), len(a)) for _ in range(draws)]
    else:
        g = np.array(groups)
        ug = sorted(set(groups))
        where = {x: np.where(g == x)[0] for x in ug}
        idx = [np.concatenate([where[x] for x in rng.choice(ug, len(ug))]) for _ in range(draws)]
    d = np.array([a[i].mean() - b[i].mean() for i in idx])
    lo = (1 - level) / 2
    return {"diff": float(a.mean() - b.mean()), "low": float(np.quantile(d, lo)), "high": float(np.quantile(d, 1 - lo))}
