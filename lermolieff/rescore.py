"""Recompute the published scores from the files in resultats/ (no GPU, no private data).

Usage: python -m lermolieff.rescore icai
- icai: the 788 test labels are read from the pinned dataset revision (the test shard, in its row order: index 0..787 of every
  prediction list); the fine-tuned VLM rows use the per-image outputs published by their author (davanstrien/iconclass-eval-preds,
  pinned revision, sha256 checked), greedy codes, invalid JSON counted as no prediction.
The external test (Rijksmuseum) will be added with its frozen list, truth and predictions.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from .measure import exact, hier_f1, paired_bootstrap

HERE = Path(__file__).resolve().parent / "resultats"  # installed with the package
DATASET, REVISION = "davanstrien/iconclass-vlm-brillfull", "8045366ff092a06e8e1c4e47f1eb43cb2a54d732"
PREDS, PREDS_REVISION = "davanstrien/iconclass-eval-preds", "a480b42cc9f6febe6952206adf55470f8e002482"
PREDS_SHA = {"passk/sft4b_1ep.json": "403ff638e4a43b37f14136ab2fc4cb8bc91b66ba770ff3d710a53559d19ff1a7",
             "passk/sft9b_1ep.json": "ee797a4c39a9e61804aec4598635a87e5821aaf232d7885c44582125e042ed4b"}


def report(gt, systems, groups=None, reference="c-radio"):
    out = {}
    scores = {n: np.array([hier_f1(g, p) for g, p in zip(gt, preds, strict=True)]) for n, preds in systems.items()}
    for n, preds in systems.items():
        out[n] = {"hier_f1": round(float(scores[n].mean()), 4), "exact_f1": round(exact(gt, preds)["f1"], 4)}
        if n != reference:
            out[n][f"minus_{reference}"] = paired_bootstrap(scores[n], scores[reference], groups)
    return out


def icai() -> dict:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    table = pq.read_table(hf_hub_download(DATASET, "data/test-00000-of-00001.parquet", repo_type="dataset", revision=REVISION),
                          columns=["label"])
    gt = table.column("label").to_pylist()
    systems = {v: json.load(open(HERE / f"icai788_{v}_predictions.json"))["T_head"] for v in ("c-radio", "pe-spatial")}
    for f, want in PREDS_SHA.items():
        path = hf_hub_download(PREDS, f, repo_type="dataset", revision=PREDS_REVISION)
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == want, f
        recs = sorted(json.load(open(path))["records"], key=lambda r: r["index"])
        assert [r["gt_codes"] for r in recs] == gt, "the author's records are not aligned with the test shard"
        systems[Path(f).stem] = [r["greedy"]["codes"] if r["greedy"].get("valid_json") else [] for r in recs]
    assert all(len(v) == len(gt) == 788 for v in systems.values())
    return {"images": len(gt), "systems": report(gt, systems)}


if __name__ == "__main__":
    assert sys.argv[1:] in ([], ["icai"]), "usage: python -m lermolieff.rescore icai"
    print(json.dumps(icai(), indent=1))
