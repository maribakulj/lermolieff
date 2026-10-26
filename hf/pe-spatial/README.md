---
license: mit
library_name: pytorch
pipeline_tag: image-classification
datasets:
  - davanstrien/iconclass-vlm-brillfull
base_model: timm/vit_pe_spatial_large_patch14_448.fb
tags:
  - iconclass
  - art-history
  - multi-label
model-index:
  - name: lermolieff-iconclass-pe-spatial
    results:
      - task:
          type: image-classification
          name: Iconclass multi-label classification
        dataset:
          name: Iconclass AI test set (ICAI, 788 images)
          type: davanstrien/iconclass-vlm-brillfull
          split: test
          revision: 8045366ff092a06e8e1c4e47f1eb43cb2a54d732
        metrics:
          - type: f1
            name: hierarchical F1 (ancestors read from the notation)
            value: 46.7
---

# lermolieff — Iconclass head on timm/vit_pe_spatial_large_patch14_448.fb

A small head (one hidden layer of 2,048) over the frozen encoder `timm/vit_pe_spatial_large_patch14_448.fb` (revision `ac6f424`, 1,024 dimensions),
predicting Iconclass codes for images of artworks. Code, method, limits and how to recompute every score:
[github.com/maribakulj/lermolieff](https://github.com/maribakulj/lermolieff).

| | hierarchical F1, ICAI 788 test images |
|---|---|
| this head | 46.7 % |

Measure: F1 between the ancestor sets of the truth and of the prediction, read from the notation; an external test (Rijksmuseum)
will be added. Version 0.1.0 is exploratory: architecture trials are under way, and a better head may replace
this one; earlier versions stay available by revision. Exploratory results: one training run; codes outside the vocabulary (7,907 codes: those seen on at least 5
training images and their ancestors) cannot be produced directly; a code missing from the catalogue counts as an error.

```bash
pip install "lermolieff[pe-spatial] @ git+https://github.com/maribakulj/lermolieff"
hf download Ma-Ri-Ba-Ku/lermolieff-iconclass-pe-spatial head.pt --local-dir pe-spatial
python -m lermolieff.predict --head pe-spatial/head.pt image.jpg
```

`head.pt`: weights, vocabulary, threshold, encoder configuration (73,344,447 bytes, sha256 `5b9e65ad7d43b6f555d6451c892949341195e89327600b1ff2a90eb930c78ef3`; details in
`manifest.json`). Licence of the head: MIT (`LICENSE`). The encoder (Meta, Apache 2.0) is not included. This variant is fully open but clearly weaker than the C-RADIO one.
Training data: davanstrien/iconclass-vlm-brillfull (CC0 1.0), revision 8045366.

## Citation

```bibtex
@software{lermolieff_2026,
  author  = {maribakulj},
  title   = {lermolieff: Iconclass codes from a frozen image encoder and a small head},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/maribakulj/lermolieff}
}
```
