"""Encoder unit tests. Uses the production model so the test path
exercises the same code that produces production embeddings."""

from __future__ import annotations

import numpy as np
import pytest

from terra_query.core.paths import MODEL_WEIGHTS_MANIFEST
from terra_query.embed import encoder, models


@pytest.fixture(scope="module")
def production_model():
    if not MODEL_WEIGHTS_MANIFEST.exists():
        pytest.skip("weights manifest not present; run fetch_weights first")
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, preprocess, tokenizer = models.load_production(device=device)
    embed_dim = models.spec(models.PRODUCTION_MODEL_ID).embed_dim
    return model, preprocess, tokenizer, device, embed_dim


def test_to_pil_rgb_shape_dtype_contract():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(3, 32, 32), dtype=np.uint8)
    img = encoder._to_pil_rgb(arr)
    assert img.size == (32, 32)
    assert img.mode == "RGB"

    with pytest.raises(ValueError):
        encoder._to_pil_rgb(arr.astype(np.float32))
    with pytest.raises(ValueError):
        encoder._to_pil_rgb(rng.integers(0, 256, size=(4, 32, 32), dtype=np.uint8))


def test_encode_image_returns_normalized_vector(production_model):
    model, preprocess, _tokenizer, device, embed_dim = production_model
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 256, size=(3, 64, 64), dtype=np.uint8)
    emb = encoder.encode_image(model, preprocess, arr, device)
    assert emb.shape == (embed_dim,)
    assert emb.dtype == np.float32
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-4


def test_encode_text_returns_normalized_vector(production_model):
    model, _preprocess, tokenizer, device, embed_dim = production_model
    emb = encoder.encode_text(model, tokenizer, "a photograph of a forest", device)
    assert emb.shape == (embed_dim,)
    assert emb.dtype == np.float32
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-4


def test_encode_image_batch_normalized_rows(production_model):
    model, preprocess, _tokenizer, device, embed_dim = production_model
    rng = np.random.default_rng(2)
    arrs = [rng.integers(0, 256, size=(3, 64, 64), dtype=np.uint8) for _ in range(5)]
    embs = encoder.encode_image_batch(model, preprocess, arrs, device, batch_size=2)
    assert embs.shape == (5, embed_dim)
    assert embs.dtype == np.float32
    norms = np.linalg.norm(embs, axis=1)
    assert (np.abs(norms - 1.0) < 1e-4).all()


def test_encode_text_batch_normalized_rows(production_model):
    model, _preprocess, tokenizer, device, embed_dim = production_model
    texts = ["a forest", "a river", "a building", "a road"]
    embs = encoder.encode_text_batch(model, tokenizer, texts, device)
    assert embs.shape == (4, embed_dim)
    assert embs.dtype == np.float32
    norms = np.linalg.norm(embs, axis=1)
    assert (np.abs(norms - 1.0) < 1e-4).all()
