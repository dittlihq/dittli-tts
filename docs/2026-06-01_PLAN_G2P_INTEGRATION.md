# Simplifying the model & folding G2P into the graph

A concrete, code-grounded plan for the open item in
[2026-05-19_PLAN_POST_V040.md](2026-05-19_PLAN_POST_V040.md) (Lever 2 — "replace
English G2P, ideally unified") and the model-side levers in
[2026-05-27_PLAN_POST_RTEN.md](2026-05-27_PLAN_POST_RTEN.md). It proposes a
**three-step sequence**, ordered by ROI and risk, and ships a working
proof-of-concept for step 1 (`scripts/export_g2p_onnx.py`).

## Where the bytes are today (English, cold cache)

| Asset | Size | What it is |
|-------|------|-----------|
| `assets/en/cmudict.json` | **5.3 MB** | 133K-word → ARPABET lookup table |
| `assets/en/g2p_model.json` | **4.5 MB** | GRU seq2seq OOV fallback, **base64 float32 in JSON** |
| `assets/en/model.onnx` | 4.7 MB | the TTS model itself (FP32 on `develop`) |

So **English G2P (9.8 MB) is larger than the TTS model**. German is rule-based
(tiny JSON) — the problem is English-specific. Two independent axes:

- **A. Fold the G2P into the graph** → kill the 9.8 MB of G2P assets + the
  hand-rolled JS GRU.
- **B. Simplify the model itself** → Vocos vocoder + 22.05 kHz (PLAN_POST_RTEN
  Phase 2); orthogonal, summarised at the end.

## The G2P architecture we actually have

`g2p_model.json` is a **single-layer GRU encoder–decoder** (confirmed from the
weight shapes):

```
chars (29 vocab) ─embed[29,256]→ GRU(256) ─→ h_enc
h_enc → GRU-decoder(256), greedy, start=<s>(2), stop=</s>(3), ≤20 steps
        → fc[74,256]+b → argmax over 74 phonemes (ARPABET + stress)
```

Inference today runs in **pure JS** (`packages/tts-en/src/g2p_predict.js`,
~190 lines of hand-written GRU matmuls) for OOV words; in-vocab words use the
5.3 MB CMU lookup. The TTS model's input contract
(`src/dittli_tts/inference/export.py`) is `x` (phoneme IDs), `tone`,
`language`, zero-filled `bert`/`ja_bert`, and the three scales — i.e. the
encoder consumes **phoneme IDs + tones**, never characters.

---

## Step 1 — Bake the neural G2P into ONNX (no retraining) ✅ PoC included

The GRU weights already exist; nothing needs training. Re-express the model as
two tiny ONNX graphs run by the **same `onnxruntime-web/wasm` runtime** the TTS
model uses:

- `g2p_encoder.onnx`: `char_ids[T] → h_enc[256]` (embed + GRU sequence).
- `g2p_decoder_step.onnx`: `(prev_phoneme_id, h[256]) → (logits[74], h'[256])`.

The greedy argmax loop stays in ~15 lines of host code (JS/Python), replacing
the ~190-line hand-rolled GRU. Decomposing the autoregressive loop into a
single-step graph avoids ONNX `Loop`/`Scan` export fragility and keeps the
graphs trivially inspectable.

**Wins**
- **Asset:** 4.5 MB base64-JSON → ~3.3 MB FP32 ONNX → **~1.7 MB FP16** (base64
  carries a 33 % tax that the binary drops, then FP16 halves it).
- **Code:** delete the bespoke JS GRU; both G2P and TTS now go through one
  SIMD-vectorised runtime (a small per-utterance speedup for OOV words).
- **No model retraining, no quality change** — bit-identical phonemes, because
  the weights and math are unchanged.

`scripts/export_g2p_onnx.py` (this branch) builds both graphs from
`g2p_model.json`, runs the greedy loop through `onnxruntime`, and asserts the
phoneme output matches the existing JS predictor
(`g2p_predict.js`) on a word list — a parity gate identical in spirit to the
German `tests/parity` suite.

**Still leaves the CMU dict (5.3 MB).** Pair with **Lever 2/D** (frequency-trim
CMU to the top ~30 K words, neural fallback handles the tail): 5.3 MB → ~1.5 MB
at <1 % OOV on a held-out corpus. After step 1 + the trim, English G2P drops
from **9.8 MB → ~3.2 MB** with zero model changes.

---

## Step 2 — Unified character front-end, distilled into the model (retraining)

This is the real simplification: make the model take **characters**, deleting
the G2P assets entirely. Per PLAN_POST_V040 Lever 2/B, prepend a small module
to `PhonemeEncoder` (`src/dittli_tts/models/synthesizer.py`):

```
chars → [char-embed + 1–2 transformer blocks] → phoneme-shaped latent →
        existing PhonemeEncoder → flow → duration → vocoder   (all frozen)
```

Train **only** the new front-end to reproduce the *current* G2P+encoder's
intermediate activations (teacher = today's pipeline). Knowledge distillation,
no audio loss in this stage — training cost is dominated by cheap forward
passes, and the existing trainer warm-starts from `init_g_ckpt`.

**The tone problem (the crux).** The encoder also consumes `tone` (ARPA stress,
4 EN levels) and `language` IDs. Two viable handlings:
1. **Auxiliary tone head**: the front-end predicts tone embeddings too,
   supervised by the current G2P's tone output. Keeps the encoder's
   tone-conditioning intact. *(recommended — lowest regression risk.)*
2. **Absorb tone into the latent**: distill straight to the post-tone-add
   activation, dropping explicit tone IDs at inference. Simpler graph, but the
   encoder's separable tone structure is lost (risk for the CN-derived symbols
   already in the table).

**Wins:** `cmudict.json` + `g2p_model.json` → **0 MB**; the front-end adds
~200–500 KB to `model.onnx`. The pack's `g2p.prepare()` hook
(`packages/tts-core/src/engine.js`) becomes a no-op, so the pack ships just
`{metadata, model}`. Same treatment later retires the German rule engine.

**Risk:** medium. Distillation is well-understood and falls back to end-to-end
fine-tuning if activation-matching plateaus; the tone head is the part to
validate first. Gate on: distill loss → near-zero on a held-out batch, then a
blind MOS test (n ≥ 20) vs the current pipeline.

---

## Step 3 (orthogonal) — simplify the model itself

From PLAN_POST_RTEN Phase 2, independent of G2P:
- Replace the HiFi-GAN `WaveformDecoder` (60–80 % of inference cost) with a
  **Vocos ISTFT head** → ~10× vocoder speedup, model 4.6 → ~2.5 MB FP16.
- Drop output rate **44.1 → 22.05 kHz** (LJSpeech is natively 22.05 kHz) →
  halves the vocoder's per-second compute for free.

Best folded into the **same fine-tune** as Step 2 (one retraining pass: new
char front-end + new vocoder head, encoder/flow/duration warm-started).

---

## Sequencing & end state

| Step | Retrain? | English G2P | model.onnx | Risk |
|------|----------|-------------|------------|------|
| today | – | 9.8 MB | 4.7 MB FP32 | – |
| **1** (G2P→ONNX + CMU trim) | no | **~3.2 MB** | 4.7 MB | low |
| **2** (unified char front-end) | yes (front-end only) | **~0 MB** | +0.3 MB | med |
| **3** (Vocos + 22 kHz) | same pass | – | **~2.5 MB** | med |

End state: a language pack is `{metadata, model.onnx}` at **~3 MB total**, one
runtime, one graph, no bespoke G2P JS and no multi-megabyte dictionaries.

## Validation gates

- **Step 1:** `scripts/export_g2p_onnx.py --verify` — ONNX greedy decode must
  match `g2p_predict.js` phoneme-for-phoneme on the test word list; measure the
  exported byte size; confirm the browser smoke app still synthesises.
- **CMU trim:** OOV rate <1 % on LJSpeech transcripts + Wikipedia leads.
- **Step 2/3:** distill loss → ~0 on held-out; blind MOS within 0.1 of current,
  n ≥ 20; parity of the char front-end Python↔JS (or move the front-end fully
  into the graph so there is no JS to keep in parity).
