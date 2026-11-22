"""Visual similarity with a frozen SigLIP model (name in config.CLIP_MODEL).
 - embed():           images -> unit-length embedding vectors (also the input of the neural-head model)
 - similarity():      raw cosine similarity of two images (a feature for the trained scorer)
 - clip_score():      the same cosine rescaled to 0-1 with fixed CLIP_LOW/HIGH (the old hand-tuned critic)
"""
import io
from typing import List

from shared import config

_cache = {}


def _load():
    if not _cache:
        # AutoImageProcessor (not AutoProcessor): we only need the image side, so no text tokenizer is required
        from transformers import AutoImageProcessor, AutoModel

        _cache["proc"] = AutoImageProcessor.from_pretrained(config.CLIP_MODEL, revision=config.CLIP_REVISION)
        _cache["model"] = AutoModel.from_pretrained(config.CLIP_MODEL, revision=config.CLIP_REVISION).eval()
    return _cache["proc"], _cache["model"]


def embed(images: List[bytes]):
    """Return a (n, d) tensor of unit-length SigLIP embeddings, one row per image."""
    import torch
    from PIL import Image

    proc, model = _load()
    imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in images]
    inputs = proc(images=imgs, return_tensors="pt")
    with torch.no_grad():
        f = model.get_image_features(pixel_values=inputs["pixel_values"])
    if not torch.is_tensor(f):  # transformers 5 returns an output object, older versions a tensor
        f = f.pooler_output
    return f / f.norm(dim=-1, keepdim=True)


def similarity(sketch: bytes, screenshot: bytes) -> float:
    """Raw cosine similarity (-1..1) between the sketch and the screenshot."""
    f = embed([sketch, screenshot])
    return float((f[0] * f[1]).sum())


def clip_score(sketch: bytes, screenshot: bytes) -> float:
    # raw cosine between a photo of a sketch and a clean render is never ~1.0, so it is
    # rescaled with fixed limits. The trained scorer replaces this guess.
    cos = similarity(sketch, screenshot)
    return max(0.0, min(1.0, (cos - config.CLIP_LOW) / (config.CLIP_HIGH - config.CLIP_LOW)))
