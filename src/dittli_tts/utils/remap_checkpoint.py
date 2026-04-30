"""Remap a phoneme embedding row across a symbol-table change.

The combined `symbols` list in dittli_tts.text.symbols is built as
`sorted(set(...))`, so adding any new IPA symbol shifts the integer ID of
every later symbol. When fine-tuning from a checkpoint trained against the
old symbol list, naive `load_state_dict` either errors on the size mismatch
or silently corrupts the embedding because row N now means a different
phoneme.

This module loads such a checkpoint, copies each row to the new index that
matches its symbol string, and randomly initializes any rows that correspond
to genuinely new symbols.
"""

from typing import Sequence

import torch
from torch import nn

PHONEME_EMB_KEY = "enc_p.emb.weight"


def build_index_map(old_symbols: Sequence[str], new_symbols: Sequence[str]) -> dict[int, int]:
    """Return {old_idx: new_idx} for every symbol present in both lists."""
    new_index = {s: i for i, s in enumerate(new_symbols)}
    return {
        old_idx: new_index[sym]
        for old_idx, sym in enumerate(old_symbols)
        if sym in new_index
    }


def remap_phoneme_embedding(
    old_weight: torch.Tensor,
    old_symbols: Sequence[str],
    new_symbols: Sequence[str],
    init_std: float = 0.02,
) -> torch.Tensor:
    """Build a [len(new_symbols), hidden] tensor by copying matched rows."""
    if old_weight.shape[0] != len(old_symbols):
        raise ValueError(
            f"Old embedding has {old_weight.shape[0]} rows but old_symbols has "
            f"{len(old_symbols)}. They must match."
        )
    hidden = old_weight.shape[1]
    new_weight = torch.empty(len(new_symbols), hidden, dtype=old_weight.dtype)
    nn.init.normal_(new_weight, mean=0.0, std=init_std)
    idx_map = build_index_map(old_symbols, new_symbols)
    for old_idx, new_idx in idx_map.items():
        new_weight[new_idx] = old_weight[old_idx]
    return new_weight


def remap_state_dict(
    state_dict: dict,
    old_symbols: Sequence[str],
    new_symbols: Sequence[str],
    init_std: float = 0.02,
    emb_key: str = PHONEME_EMB_KEY,
) -> dict:
    """Return a new state_dict with the phoneme embedding row-aligned to `new_symbols`.

    All other tensors are passed through unchanged.
    """
    if emb_key not in state_dict:
        raise KeyError(
            f"Expected key {emb_key!r} in checkpoint. Available keys with 'emb': "
            f"{[k for k in state_dict if 'emb' in k]}"
        )
    out = dict(state_dict)
    out[emb_key] = remap_phoneme_embedding(
        state_dict[emb_key], old_symbols, new_symbols, init_std=init_std
    )
    return out


def load_old_symbols(path: str) -> list[str]:
    """Load a saved old symbol list (one symbol per line, utf-8)."""
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def save_symbols(symbols: Sequence[str], path: str) -> None:
    """Persist a symbol list so future fine-tunes can find the old ordering."""
    with open(path, "w", encoding="utf-8") as f:
        for sym in symbols:
            f.write(sym + "\n")
