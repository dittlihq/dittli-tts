"""Tests for load_engine checkpoint loading behaviour."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


def test_load_engine_pads_short_embedding(tmp_path):
    """A checkpoint with fewer symbols than the current table should load correctly.

    The rows present in the checkpoint must be preserved exactly; the extra
    rows (new symbols) are left at their random init values.
    """
    import torch

    from dittli_tts.inference.engine import load_engine
    from dittli_tts.models.synthesizer import VoiceSynthesizer
    from dittli_tts.text.symbols import symbols
    from dittli_tts.utils.config import MODEL_PARAMS, N_SPEAKERS, SEGMENT_FRAMES, SPEC_CHANNELS

    n_current = len(symbols)
    n_old = n_current - 1  # simulate a pre-extension symbol table

    old_model = VoiceSynthesizer(n_old, SPEC_CHANNELS, SEGMENT_FRAMES, n_speakers=N_SPEAKERS, **MODEL_PARAMS)
    ckpt_path = tmp_path / "G_old.pth"
    torch.save({"model": old_model.state_dict()}, str(ckpt_path))

    engine = load_engine(str(ckpt_path), device="cpu")

    old_emb = old_model.enc_p.emb.weight.detach()
    new_emb = engine.enc_p.emb.weight.detach()

    assert new_emb.shape[0] == n_current, "loaded model must have current symbol count"
    assert torch.allclose(old_emb, new_emb[:n_old]), "rows from the checkpoint must be preserved exactly"
