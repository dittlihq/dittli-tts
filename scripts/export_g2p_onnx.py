#!/usr/bin/env python
"""Proof-of-concept: bake the English neural G2P (GRU seq2seq) into ONNX.

Step 1 of docs/2026-06-01_PLAN_G2P_INTEGRATION.md. The OOV fallback currently
ships as `packages/tts-en/assets/en/g2p_model.json` — 4.5 MB of base64 float32
in JSON — and runs in ~190 lines of hand-rolled GRU matmuls
(`packages/tts-en/src/g2p_predict.js`). This script re-expresses the *same
weights* (no retraining) as two small ONNX graphs that run on the same
onnxruntime-web/wasm runtime as the TTS model:

  g2p_encoder.onnx       char_ids[1,T] (int64) -> h_enc[1,256]
  g2p_decoder_step.onnx  prev_id[1] (int64), h[1,256] -> logits[1,74], h'[1,256]

The greedy argmax loop stays in ~15 lines of host code (here, and in JS).

Run / verify (deps are dev-only, not added to the project):

    uv run --extra dev --with onnx --with onnxruntime \
        python scripts/export_g2p_onnx.py --out /tmp/g2p --verify

`--verify` greedy-decodes a word list through onnxruntime and asserts the
phoneme output matches the `g2p_en` library these weights came from — the same
source `g2p_predict.js` ports, so parity with the library is parity with what
ships.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
G2P_JSON = REPO_ROOT / "packages" / "tts-en" / "assets" / "en" / "g2p_model.json"

HIDDEN = 256
START_ID = 2  # "<s>" in the phoneme vocab
EOS_ID = 3  # "</s>"
MAX_DECODE = 20  # matches g2p_predict.js


def _load_weights(path: Path) -> dict:
    raw = json.loads(path.read_text())
    out: dict = {}
    for k, v in raw.items():
        if isinstance(v, dict) and "shape" in v:
            flat = np.frombuffer(base64.b64decode(v["data"]), dtype=np.float32)
            out[k] = flat.reshape(v["shape"]).copy()
        else:
            out[k] = v
    return out


class Encoder(nn.Module):
    """char ids -> final GRU hidden state. Uses nn.GRU so ONNX gets one GRU op
    and variable sequence length comes for free."""

    def __init__(self, w: dict):
        super().__init__()
        self.emb = nn.Embedding(*w["enc_emb"].shape)
        self.gru = nn.GRU(HIDDEN, HIDDEN, batch_first=True)
        with torch.no_grad():
            self.emb.weight.copy_(torch.from_numpy(w["enc_emb"]))
            self.gru.weight_ih_l0.copy_(torch.from_numpy(w["enc_w_ih"]))
            self.gru.weight_hh_l0.copy_(torch.from_numpy(w["enc_w_hh"]))
            self.gru.bias_ih_l0.copy_(torch.from_numpy(w["enc_b_ih"]))
            self.gru.bias_hh_l0.copy_(torch.from_numpy(w["enc_b_hh"]))

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:  # [1,T] -> [1,256]
        _, h_n = self.gru(self.emb(char_ids))
        return h_n[0]


class DecoderStep(nn.Module):
    """One autoregressive step. GRU cell math is written out explicitly (gate
    order [r, z, n]) so it matches torch.nn.GRUCell *and* g2p_predict.js, and
    exports to elementary ONNX ops with no GRUCell-export dependency."""

    def __init__(self, w: dict):
        super().__init__()
        self.emb = nn.Embedding(*w["dec_emb"].shape)
        self.w_ih = nn.Parameter(torch.from_numpy(w["dec_w_ih"]), requires_grad=False)
        self.w_hh = nn.Parameter(torch.from_numpy(w["dec_w_hh"]), requires_grad=False)
        self.b_ih = nn.Parameter(torch.from_numpy(w["dec_b_ih"]), requires_grad=False)
        self.b_hh = nn.Parameter(torch.from_numpy(w["dec_b_hh"]), requires_grad=False)
        self.fc = nn.Linear(*reversed(w["fc_w"].shape))
        with torch.no_grad():
            self.emb.weight.copy_(torch.from_numpy(w["dec_emb"]))
            self.fc.weight.copy_(torch.from_numpy(w["fc_w"]))
            self.fc.bias.copy_(torch.from_numpy(w["fc_b"]))

    def forward(self, prev_id: torch.Tensor, h: torch.Tensor):  # [1],[1,256]
        x = self.emb(prev_id)  # [1,256]
        gi = x @ self.w_ih.T + self.b_ih  # [1,768]
        gh = h @ self.w_hh.T + self.b_hh
        i_r, i_z, i_n = gi.chunk(3, dim=1)
        h_r, h_z, h_n = gh.chunk(3, dim=1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        h_new = (1 - z) * n + z * h
        return self.fc(h_new), h_new


def _to_fp16(path: Path) -> None:
    """Convert an ONNX file to FP16 in place, keeping I/O types FP32 so the host
    loop passes plain float32 tensors across the encoder→decoder boundary."""
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    m = onnx.load(str(path))
    onnx.save(convert_float_to_float16(m, keep_io_types=True), str(path))


def export(w: dict, out_dir: Path, fp16: bool = False) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    enc_path = out_dir / "g2p_encoder.onnx"
    dec_path = out_dir / "g2p_decoder_step.onnx"

    enc = Encoder(w).eval()
    torch.onnx.export(
        enc,
        (torch.zeros(1, 5, dtype=torch.long),),
        str(enc_path),
        input_names=["char_ids"],
        output_names=["h_enc"],
        dynamic_axes={"char_ids": {1: "T"}},
        opset_version=14,
        dynamo=False,
    )

    dec = DecoderStep(w).eval()
    torch.onnx.export(
        dec,
        (torch.tensor([START_ID]), torch.zeros(1, HIDDEN)),
        str(dec_path),
        input_names=["prev_id", "h"],
        output_names=["logits", "h_out"],
        opset_version=14,
        dynamo=False,
    )

    if fp16:
        _to_fp16(enc_path)
        _to_fp16(dec_path)
    return enc_path, dec_path


def write_vocab(w: dict, out_dir: Path) -> Path:
    """Sidecar the host greedy-loop needs: grapheme→id input table, id→phoneme
    output table, and the start/eos token ids. Kept tiny and language-agnostic."""
    vocab_path = out_dir / "g2p_vocab.json"
    vocab_path.write_text(
        json.dumps(
            {
                "graphemes": w["graphemes"],
                "phonemes": w["phonemes"],
                "start_id": START_ID,
                "eos_id": EOS_ID,
                "max_decode": MAX_DECODE,
            }
        )
    )
    return vocab_path


def greedy_decode_onnx(word: str, enc_sess, dec_sess, g2idx, phonemes) -> list[str]:
    chars = list(word) + ["</s>"]
    ids = np.array([[g2idx.get(c, 1) for c in chars]], dtype=np.int64)
    h = enc_sess.run(["h_enc"], {"char_ids": ids})[0]
    prev = np.array([START_ID], dtype=np.int64)
    out: list[str] = []
    for _ in range(MAX_DECODE):
        logits, h = dec_sess.run(["logits", "h_out"], {"prev_id": prev, "h": h})
        idx = int(logits[0].argmax())
        if idx == EOS_ID:
            break
        out.append(phonemes[idx])
        prev = np.array([idx], dtype=np.int64)
    return out


def verify(enc_path: Path, dec_path: Path, w: dict) -> int:
    import onnxruntime as ort
    from g2p_en import G2p

    g2idx = {g: i for i, g in enumerate(w["graphemes"])}
    phonemes = w["phonemes"]
    enc_sess = ort.InferenceSession(str(enc_path), providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(str(dec_path), providers=["CPUExecutionProvider"])

    g2p = G2p()
    # OOV-ish words that exercise the neural fallback (not the CMU lookup).
    words = [
        "blorptang",
        "zylophonics",
        "qwertzuiop",
        "schmenkle",
        "kubernetes",
        "frumiousbandersnatch",
        "tensorflow",
        "dittli",
        "vauxen",
        "glorptastic",
    ]
    mismatches = 0
    for word in words:
        got = greedy_decode_onnx(word, enc_sess, dec_sess, g2idx, phonemes)
        ref = g2p.predict(word)
        status = "ok" if got == ref else "MISMATCH"
        if got != ref:
            mismatches += 1
        print(f"  [{status}] {word:24} onnx={got}\n{'':30}g2p_en={ref}")
    print(f"\n{len(words) - mismatches}/{len(words)} words match g2p_en.")
    return 1 if mismatches else 0


EN_ASSETS = REPO_ROOT / "packages" / "tts-en" / "assets" / "en"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/g2p_onnx", help="output dir for the ONNX graphs")
    ap.add_argument(
        "--assets",
        action="store_true",
        help=f"write the ship-able assets into {EN_ASSETS} instead of --out",
    )
    ap.add_argument("--fp16", action="store_true", help="convert the graphs to FP16")
    ap.add_argument("--verify", action="store_true", help="check parity against g2p_en")
    args = ap.parse_args()

    out_dir = EN_ASSETS if args.assets else Path(args.out)
    w = _load_weights(G2P_JSON)
    enc_path, dec_path = export(w, out_dir, fp16=args.fp16)
    vocab_path = write_vocab(w, out_dir)
    enc_mb = os.path.getsize(enc_path) / 1e6
    dec_mb = os.path.getsize(dec_path) / 1e6
    vocab_kb = os.path.getsize(vocab_path) / 1e3
    json_mb = os.path.getsize(G2P_JSON) / 1e6
    kind = "FP16" if args.fp16 else "FP32"
    print(f"wrote {enc_path.name} ({enc_mb:.2f} MB) + {dec_path.name} ({dec_mb:.2f} MB)")
    print(f"      {vocab_path.name} ({vocab_kb:.1f} KB)  [{kind}]")
    print(f"ONNX total: {enc_mb + dec_mb:.2f} MB  vs  base64 JSON today: {json_mb:.2f} MB")

    if args.verify:
        return verify(enc_path, dec_path, w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
