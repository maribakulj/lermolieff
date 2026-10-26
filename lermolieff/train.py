"""Retrain a head from scratch: encode the pinned dataset, split, train, choose the epoch and threshold on validation, measure.

Usage: python -m lermolieff.train --encoder c-radio|pe-spatial --out DIR [--limit N]
Needs a GPU (about 2 h on a T4 for the encoding). Features are cached per shard in DIR/feats with a manifest; a resumed run reuses
them only if the manifest matches. --limit keeps N rows per shard for a dry run (results are then technical only).
Writes DIR/head.pt, DIR/journal.jsonl and DIR/resultats.json (hierarchical F1 and exact codes on the 788 test images).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import io
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .encoders import feature_config, load_encoder
from .head import asym_loss, build_vocab, make_head, targets
from .hierarchy import most_specific
from .measure import exact, per_image

DATASET, REVISION = "davanstrien/iconclass-vlm-brillfull", "8045366ff092a06e8e1c4e47f1eb43cb2a54d732"
TEST = "data/test-00000-of-00001.parquet"
TRAIN = [f"data/train-{i:05d}-of-00011.parquet" for i in range(11)]
GRID = [round(float(x), 2) for x in np.arange(0.05, 0.61, 0.05)]
EPOCHS, BATCH, LR, WD = 15, 512, 1e-3, 1e-4


def version(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def encode(shard, features, cfg, out: Path, limit: int):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    tag = Path(shard).stem
    npz, man = out / "feats" / f"{tag}.npz", out / "feats" / f"{tag}.json"
    table = pq.read_table(hf_hub_download(DATASET, shard, repo_type="dataset", revision=REVISION))
    n = table.num_rows if not limit else min(limit, table.num_rows)
    labels = table.column("label").to_pylist()[:n]
    manifest = {"dataset": [DATASET, REVISION], "shard": shard, "rows": n, "limit": limit, "feature_config": cfg,
                "labels_sha256": hashlib.sha256(json.dumps(labels).encode()).hexdigest()}
    if npz.exists() and man.exists() and json.load(open(man)) == manifest:
        z = np.load(npz)
        assert z["x"].shape[0] == n == len(z["md5"]) and np.isfinite(z["x"]).all(), f"{tag}: cache does not match"
        return z["x"], labels, list(z["md5"])
    imgs = table.column("image").to_pylist()[:n]
    xs, md5 = [], []
    for i in range(0, n, 32):
        chunk = imgs[i:i + 32]
        md5 += [hashlib.md5(x["bytes"]).hexdigest() for x in chunk]
        xs.append(features([Image.open(io.BytesIO(x["bytes"])) for x in chunk]))
    x = np.concatenate(xs)
    assert x.shape[0] == n and np.isfinite(x).all()
    np.savez(out / "feats" / f"{tag}.tmp.npz", x=x, md5=np.array(md5))
    os.replace(out / "feats" / f"{tag}.tmp.npz", npz)
    json.dump(manifest, open(out / "feats" / f"{tag}.tmp.json", "w"))
    os.replace(out / "feats" / f"{tag}.tmp.json", man)
    return x, labels, md5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["c-radio", "pe-spatial"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    (a.out / "feats").mkdir(parents=True, exist_ok=True)
    cfg = feature_config(a.encoder)
    features = load_encoder(a.encoder, dev)

    XT, YT, _ = encode(TEST, features, cfg, a.out, a.limit)
    parts = [encode(s, features, cfg, a.out, a.limit) for s in TRAIN]
    X = np.concatenate([p[0] for p in parts]).astype(np.float32)
    Y = [y for p in parts for y in p[1]]
    M = [m for p in parts for m in p[2]]

    # Validation: 5 % of the train, drawn by groups of byte-identical images (seed 0).
    groups = sorted(set(M))
    gval = set(np.array(groups)[np.random.default_rng(0).random(len(groups)) < 0.05])
    val = np.array([m in gval for m in M])
    Xtr, Xva = X[~val], X[val]
    Ytr, Yva = [y for y, v in zip(Y, val) if not v], [y for y, v in zip(Y, val) if v]

    vocab = build_vocab(Ytr)
    Ttr = torch.from_numpy(targets(Ytr, vocab))
    head = make_head(X.shape[1], len(vocab)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)

    @torch.no_grad()
    def probs(Q):
        head.eval()
        return torch.cat([torch.sigmoid(head(torch.from_numpy(Q[i:i + 4096]).to(dev))) for i in range(0, len(Q), 4096)]).cpu().numpy()

    def pred(P, t):
        return [most_specific([vocab[j] for j in np.nonzero(row >= t)[0]]) for row in P]

    Xtr_t, gen = torch.from_numpy(Xtr), torch.Generator().manual_seed(0)
    best = (-1.0, None, None)
    for epoch in range(EPOCHS):
        head.train()
        perm = torch.randperm(len(Xtr_t), generator=gen)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            loss = asym_loss(head(Xtr_t[b].to(dev)), Ttr[b].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        Pva = probs(Xva)
        t = max(GRID, key=lambda t: per_image(Yva, pred(Pva, t)).mean())
        score = float(per_image(Yva, pred(Pva, t)).mean())
        with open(a.out / "journal.jsonl", "a") as f:
            f.write(json.dumps({"epoch": epoch, "val_hier_f1": score, "threshold": t, "loss": float(loss)}) + "\n")
        if score > best[0]:
            best = (score, epoch, t)
            torch.save({"state": head.state_dict(), "vocab": vocab, "threshold": float(t), "epoch": epoch,
                        "feature_config": cfg}, a.out / "head.pt")
    head.load_state_dict(torch.load(a.out / "head.pt", weights_only=True)["state"])
    PT = pred(probs(XT.astype(np.float32)), best[2])
    res = {"technical_only": bool(a.limit), "feature_config": cfg, "dataset": [DATASET, REVISION],
           "versions": {p: version(p) for p in ("lermolieff", "torch", "transformers", "timm", "open-clip-torch", "numpy")},
           "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "validation_md5_sha256": hashlib.sha256(json.dumps(sorted(gval)).encode()).hexdigest(), "device": dev,
           "best_epoch": best[1], "threshold": best[2], "val_hier_f1": best[0], "vocab": len(vocab),
           "test": {"images": len(YT), "hier_f1": float(per_image(YT, PT).mean()), "exact": exact(YT, PT)}}
    json.dump(res, open(a.out / "resultats.json", "w"), indent=1)
    json.dump(PT, open(a.out / "predictions.json", "w"))
    print(json.dumps(res["test"], indent=1))


if __name__ == "__main__":
    main()
