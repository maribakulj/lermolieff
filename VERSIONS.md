# Tested environment

End-to-end check (image → codes) of version 0.1.0 (the code of this release), installed with `pip install ".[c-radio,pe-spatial,train]"` in a fresh Kaggle
environment (Python 3.13.15, NVIDIA Tesla T4), 25/10/2026: for the first 32 test images of
`davanstrien/iconclass-vlm-brillfull` (revision `8045366`), both variants return exactly the published codes (32/32 each).

| package | version |
|---|---|
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| transformers | 5.16.1 |
| timm | 1.0.29 |
| open_clip_torch | 3.3.0 |
| einops | 0.8.2 |
| numpy | 2.1.3 |
| pillow | 13.0.0 |
| huggingface_hub | 1.29.0 |
| pyarrow | 23.0.1 |
| safetensors | 0.8.0 |
