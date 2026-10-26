# lermolieff

Iconclass codes for images of artworks, from a frozen image encoder and a small trained head. Two variants:

| variant | encoder (not included, downloaded from Hugging Face) | head weights | licence of the encoder |
|---|---|---|---|
| **c-radio** (recommended) | `nvidia/C-RADIOv3-L` @ `9d04134` | [`Ma-Ri-Ba-Ku/lermolieff-iconclass-c-radio`](https://huggingface.co/Ma-Ri-Ba-Ku/lermolieff-iconclass-c-radio) | NVIDIA Open Model License |
| **pe-spatial** (fully open) | `timm/vit_pe_spatial_large_patch14_448.fb` @ `ac6f424` | [`Ma-Ri-Ba-Ku/lermolieff-iconclass-pe-spatial`](https://huggingface.co/Ma-Ri-Ba-Ku/lermolieff-iconclass-pe-spatial) | Apache 2.0 |

## Results

Hierarchical F1 by the notation: the truth and the prediction are each closed under their ancestors, read from the notation itself
(`lermolieff/hierarchy.py`); F1 between the two sets, averaged over images. Test set: the 788 test images of
[`davanstrien/iconclass-vlm-brillfull`](https://huggingface.co/datasets/davanstrien/iconclass-vlm-brillfull) (revision `8045366`).

| | ICAI 788 | exact codes F1 | seconds per image (T4, batch 1) |
|---|---|---|---|
| head, c-radio | **54.8 %** | 0.275 | 0.058 |
| head, pe-spatial | 46.7 % | 0.203 | 0.074 |
| Qwen3.5-9B fine-tuned by van Strien ¹ | 48.3 % | 0.262 | — |
| Qwen3.5-4B fine-tuned by van Strien ¹ | 44.8 % | 0.226 | 2.12 ² |

¹ The per-image greedy outputs published by the author ([`davanstrien/iconclass-eval-preds`](https://huggingface.co/datasets/davanstrien/iconclass-eval-preds),
revision `a480b42`, `passk/sft*_1ep.json`), re-scored with the measure above; these are not the author's own figures, which use another
measure. ² Our run of the published 4B checkpoint on 50 ICAI test images; its outputs matched the published ones on only 62 % of those
images, so this time is indicative of the model, not of the exact run behind the published outputs.

Differences with 95 % intervals (bootstrap over images): pe-spatial − c-radio = −8.0 [−9.2; −6.9] points; c-radio − 9B = +6.5
[+5.1; +7.9]. The rule fixed before computing pe-spatial was to recommend it only if its lower bound stayed above −2 points: c-radio
is recommended. Every figure is recomputed by `python -m lermolieff.rescore icai` from the files installed with the package
(`lermolieff/resultats/`), the pinned test labels and the author's published outputs (sha256 checked). Per-branch figures:
`lermolieff/resultats/par_usage.json`.

**External test.** A test outside the training corpus (Rijksmuseum paintings, prints and drawings, open data), with a frozen list,
truth and predictions that anyone can rescore, is being built and will be added before the first release.

## Limits

- The vocabulary holds the codes seen on at least 5 training images and their ancestors (7,907 codes); other codes cannot be produced
  directly (10.1 % of the ICAI test codes, 349 of 3,456).
- A code missing from the catalogue counts as an error, even when the image shows it.
- One training run, one seed. Epoch and threshold were chosen on validation, but the test set also served to compare systems and to
  choose between encoders: these results are exploratory.

## Install and use

```bash
pip install "lermolieff[c-radio] @ git+https://github.com/maribakulj/lermolieff"      # or [pe-spatial]
hf download Ma-Ri-Ba-Ku/lermolieff-iconclass-c-radio head.pt --local-dir c-radio     # 98.5 MB, sha256 2b403a91…
python -m lermolieff.predict --head c-radio/head.pt image1.jpg image2.jpg
```

The versions checked end to end (image to codes, both variants, on a fresh remote machine) are listed in `VERSIONS.md`. The head file names the encoder it was trained on; `predict` loads that encoder. In Python:

```python
from PIL import Image
from lermolieff.encoders import load_encoder
from lermolieff.head import Head

head = Head("c-radio/head.pt", "cuda")
features = load_encoder("c-radio", "cuda")
codes = head.predict(features([Image.open("image.jpg")]).astype("float32"))[0]
```

## Method

- Encoder frozen; image padded to a square on grey, 448 px. c-radio: summary token and mean of the spatial tokens, each normalised,
  concatenated (4,096 dimensions); pe-spatial: mean of the patch tokens (1,024).
- Head: one hidden layer of 2,048 (GELU, dropout 0.2), one output per vocabulary code; targets closed under ancestors; asymmetric
  loss (γ− = 4, γ+ = 0, clip 0.05); AdamW, lr 1e-3, weight decay 1e-4, batch 512, 15 epochs.
- Epoch and threshold chosen on a validation set of 5 % of the training images, drawn by groups of byte-identical images (seed 0).
  Prediction: codes above the threshold, ancestors implied by another code removed.
- Retrain: `pip install "lermolieff[c-radio,train] @ git+…"`, then `python -m lermolieff.train --encoder c-radio --out run/` (GPU,
  about 2 h on a T4). This rewrites the recipe; a retrained head is not guaranteed to be numerically identical to the published one.

## Licence

Code: MIT (`LICENSE`). Head weights: MIT for both variants. The encoders are not redistributed: using the c-radio variant means
using C-RADIOv3-L under the NVIDIA Open Model License (`NOTICE`, `LICENSE-NVIDIA-OPEN-MODEL.txt`); pe-spatial uses a model under
Apache 2.0. Training data: CC0 1.0.
