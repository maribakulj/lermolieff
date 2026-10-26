"""The two frozen image encoders, pinned to a revision, and their feature functions (image list -> L2-normalised float16 vectors)."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

ENCODERS = {
    "c-radio": ("nvidia/C-RADIOv3-L", "9d0413465e8a91e67bbf2c1ad342815478d1b906",
                "|square-grey128|448|bicubic|autocast-fp16|summary+meanspatial|l2-each-then-l2"),
    "pe-spatial": ("timm/vit_pe_spatial_large_patch14_448.fb", "ac6f424680a305470340dac75a0a9b4c47ee030f",
                   "|square-grey128|448|bicubic|autocast-fp16|meanpatch|l2"),
}


def feature_config(name: str) -> str:
    return "{}@{}{}".format(*ENCODERS[name])


def square(im: Image.Image) -> Image.Image:
    """Pad to a square on mid grey, then 448 px bicubic."""
    im = im.convert("RGB")
    s = max(im.size)
    c = Image.new("RGB", (s, s), (128, 128, 128))
    c.paste(im, ((s - im.width) // 2, (s - im.height) // 2))
    return c.resize((448, 448), Image.BICUBIC)


def load_encoder(name: str, device: str):
    repo, rev, _ = ENCODERS[name]
    if name == "c-radio":
        from transformers import AutoModel, CLIPImageProcessor

        model = AutoModel.from_pretrained(repo, revision=rev, trust_remote_code=True).to(device).eval()
        processor = CLIPImageProcessor.from_pretrained(repo, revision=rev)

        @torch.no_grad()
        def features(images):
            px = processor(images=[square(i) for i in images], return_tensors="pt", do_resize=False).pixel_values.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):  # RADIO normalises its input itself
                summary, spatial = model(px)
            v = torch.cat([torch.nn.functional.normalize(summary.float(), dim=-1),
                           torch.nn.functional.normalize(spatial.float().mean(1), dim=-1)], 1)
            return torch.nn.functional.normalize(v, dim=-1).cpu().numpy().astype(np.float16)

        return features

    import timm
    import torchvision.transforms as T
    from huggingface_hub import hf_hub_download

    weights = hf_hub_download(repo, "model.safetensors", revision=rev)
    model = timm.create_model("vit_pe_spatial_large_patch14_448", pretrained=True,
                              pretrained_cfg_overlay={"file": weights}).to(device).eval()
    to_t = T.Compose([T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])  # the model's own mean and std

    @torch.no_grad()
    def features(images):
        x = torch.stack([to_t(square(i)) for i in images]).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            tokens = model.forward_features(x)
        v = tokens[:, getattr(model, "num_prefix_tokens", 0):].float().mean(1)  # mean of the patch tokens (timm's own pooling)
        return torch.nn.functional.normalize(v, dim=-1).cpu().numpy().astype(np.float16)

    return features
