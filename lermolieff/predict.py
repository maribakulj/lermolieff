"""Iconclass codes for images.

Usage: python -m lermolieff.predict --head head.pt IMAGE [IMAGE ...]
The encoder is the one the head was trained on (read from head.pt). Prints one JSON line per image: file, codes.
"""

from __future__ import annotations

import argparse
import json

import torch
from PIL import Image

from .encoders import ENCODERS, feature_config, load_encoder
from .head import Head


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--threshold", type=float)
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    head = Head(a.head, dev)
    name = next(n for n in ENCODERS if feature_config(n) == head.feature_config)
    features = load_encoder(name, dev)
    for i in range(0, len(a.images), 16):
        files = a.images[i:i + 16]
        codes = head.predict(features([Image.open(f) for f in files]).astype("float32"), a.threshold)
        for f, c in zip(files, codes, strict=True):
            print(json.dumps({"file": f, "codes": c}, ensure_ascii=False))


if __name__ == "__main__":
    main()
