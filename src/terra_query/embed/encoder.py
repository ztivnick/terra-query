"""Image and text encoding helpers. Pure model I/O.

No filesystem, no chip iteration. Returns L2-normalized float32 tensors
(or numpy arrays) so cosine similarity is a plain dot product downstream.
"""

from __future__ import annotations


def _to_pil_rgb(arr):
    """Convert a (bands, h, w) uint8 numpy array to a (h, w, 3) PIL Image.

    The chip is read by `read_chip` as (n_bands, native_h, native_w). Both
    RGB (bands=(1,2,3)) and CIR (bands=(4,1,2)) come out as 3-band arrays
    we want to feed into a CLIP-style ImageNet preprocessor (PIL -> tensor).
    PIL does not care semantically whether the 3 channels are R/G/B or
    NIR/R/G; it just packs them as a 3-channel image.
    """
    import numpy as np
    from PIL import Image

    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"expected (3, h, w) uint8 array, got {arr.shape} {arr.dtype}")
    if arr.dtype != np.uint8:
        raise ValueError(f"expected uint8, got {arr.dtype}")
    hwc = np.transpose(arr, (1, 2, 0))  # (h, w, 3)
    return Image.fromarray(hwc, mode="RGB")


def encode_image(model, preprocess, image_or_arr, device) -> "np.ndarray":
    """Encode one image -> (embed_dim,) L2-normalized float32 numpy array.

    `image_or_arr` is either a PIL.Image or a numpy (3, h, w) uint8 array
    (the shape returned by `read_chip`).
    """
    import numpy as np
    import torch
    from PIL import Image

    if isinstance(image_or_arr, np.ndarray):
        img = _to_pil_rgb(image_or_arr)
    elif isinstance(image_or_arr, Image.Image):
        img = image_or_arr
    else:
        raise TypeError(f"image_or_arr must be PIL.Image or ndarray, got {type(image_or_arr)}")

    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0).cpu().float().numpy()


def encode_text(model, tokenizer, text: str, device) -> "np.ndarray":
    """Encode one text string -> (embed_dim,) L2-normalized float32 numpy array."""
    import torch

    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0).cpu().float().numpy()


def encode_image_batch(model, preprocess, images, device, batch_size: int) -> "np.ndarray":
    """Encode a list of images -> (N, embed_dim) L2-normalized float32 numpy array.

    `images` is a list/iterable of (3, h, w) uint8 ndarrays or PIL Images.
    Batches at `batch_size` to keep MPS memory in check.
    """
    import numpy as np
    import torch
    from PIL import Image

    out_chunks: list[np.ndarray] = []
    batch_tensors: list[torch.Tensor] = []

    def _flush(buf):
        if not buf:
            return
        x = torch.stack(buf, dim=0).to(device)
        with torch.no_grad():
            emb = model.encode_image(x)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        out_chunks.append(emb.cpu().float().numpy())

    for img in images:
        if isinstance(img, np.ndarray):
            img = _to_pil_rgb(img)
        elif not isinstance(img, Image.Image):
            raise TypeError(f"each image must be PIL.Image or ndarray, got {type(img)}")
        batch_tensors.append(preprocess(img))
        if len(batch_tensors) >= batch_size:
            _flush(batch_tensors)
            batch_tensors = []
    _flush(batch_tensors)

    if not out_chunks:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(out_chunks, axis=0)


def encode_text_batch(model, tokenizer, texts, device) -> "np.ndarray":
    """Encode a list of text strings -> (N, embed_dim) L2-normalized float32 array."""
    import torch

    tokens = tokenizer(list(texts)).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().float().numpy()
