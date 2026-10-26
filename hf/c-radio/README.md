---
license: mit
library_name: pytorch
pipeline_tag: image-classification
datasets:
  - davanstrien/iconclass-vlm-brillfull
base_model: nvidia/C-RADIOv3-L
tags:
  - iconclass
  - art-history
  - multi-label
model-index:
  - name: lermolieff-iconclass-c-radio
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
            value: 54.8
---

# lermolieff — Iconclass head on nvidia/C-RADIOv3-L

A small head (one hidden layer of 2,048) over the frozen encoder `nvidia/C-RADIOv3-L` (revision `9d04134`, 4,096 dimensions),
predicting Iconclass codes for images of artworks. Code, method, limits and how to recompute every score:
[github.com/maribakulj/lermolieff](https://github.com/maribakulj/lermolieff).

| | hierarchical F1, ICAI 788 test images |
|---|---|
| this head | 54.8 % |

Measure: F1 between the ancestor sets of the truth and of the prediction, read from the notation; an external test (Rijksmuseum)
will be added. Version 0.1.0 is exploratory: architecture trials are under way, and a better head may replace
this one; earlier versions stay available by revision. Exploratory results: one training run; codes outside the vocabulary (7,907 codes: those seen on at least 5
training images and their ancestors) cannot be produced directly; a code missing from the catalogue counts as an error.

```bash
pip install "lermolieff[c-radio] @ git+https://github.com/maribakulj/lermolieff"
hf download Ma-Ri-Ba-Ku/lermolieff-iconclass-c-radio head.pt --local-dir c-radio
python -m lermolieff.predict --head c-radio/head.pt image.jpg
```

`head.pt`: weights, vocabulary, threshold, encoder configuration (98,510,271 bytes, sha256 `2b403a912383853d52ed771ef5843a42a80160e39b180ccf98e8bf3b1fc98242`; details in
`manifest.json`). Licence of the head: MIT (`LICENSE`). The encoder is not included. Using this head means using C-RADIOv3-L under the NVIDIA Open Model License (`NOTICE` and `LICENSE-NVIDIA-OPEN-MODEL.txt` here): Licensed by NVIDIA Corporation under the NVIDIA Open Model License.
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
