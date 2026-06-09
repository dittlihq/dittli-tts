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
- **Reusable across languages:** build the host greedy-loop
  **language-agnostically** (a generic `encoder.onnx` + `decoder_step.onnx`
  driver, parameterised by the pack's grapheme/phoneme tables) rather than as
  English-only glue. Any future language that wants a neural G2P then ships only
  its own weights — no new hand-ported JS, no new parity suite. This is a
  deliberate choice to keep step 1 friendly to contributors (see "Impact on
  adding new languages" below).
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

---

## Impact on adding new languages

The architecture stays **per-language packs** throughout (a separate
`model.onnx` per language) — none of these steps imply or require a single
shared multilingual model; that is a separate, larger decision.

### What adding a language costs today

German required: (a) phoneme symbols in the shared table; (b) a G2P written
**twice** — Python (`german.py`) *and* a hand-ported JS copy (`g2p_de.js`);
(c) a generated rules JSON; (d) an **805-word Python↔JS parity suite** to keep
the two in sync; (e) tones / language IDs; (f) a fine-tuned checkpoint; (g) a
shipped pack `{metadata, model, g2p assets}`. The brittle, expensive parts are
(b) + (d): the duplicated implementation and its parity test.

### How each step changes that

- **Step 1 — neutral-to-positive, and reusable.** The English-specific parts
  (CMU trim, the GRU weights) don't touch other languages, but the
  **generic `encoder.onnx` + `decoder_step.onnx` + host greedy-loop becomes
  language-agnostic G2P infrastructure.** A future language whose orthography is
  irregular enough to want a neural G2P ships only its weights through that same
  loop — removing the JS-port and parity-suite tax for those languages, without
  forcing anything on rule-based ones.

- **Step 2 — the big lever, and it cuts both ways.**
  - *Win:* a pack becomes `{metadata, model.onnx}` — **no shipped G2P, no JS
    port, no parity suite.** Items (b), (c), (d) and the G2P assets in (g) all
    disappear. This is the single largest reduction in per-language
    *engineering/maintenance* effort.
  - *The catch:* the front-end is trained by **distilling against a teacher G2P
    for that language**, so real linguistic G2P knowledge is still required —
    but only **at training time, never at ship time**. For a brand-new language
    you first build a teacher (rules, or borrow `espeak-ng` / `phonemizer`)
    purely to generate targets; it never ships and never needs a JS twin. The
    cost **moves from "maintain two G2P implementations forever" to "have a G2P
    once, during training."** "G2P-free packs" is not "no linguistic work."
  - *Bar shifts from engineering to ML:* adding a language now **requires a
    training run** (distillation + a blind listening check) rather than "write
    rules + fine-tune." Tone-bearing or phonologically rich languages lean
    harder on data quality; rule-based additions are more predictable than
    char-level ones.

- **Step 3 — language-agnostic.** Vocoder / sample-rate is about audio
  synthesis, not text. No effect on the add-language flow beyond new fine-tunes
  targeting the new vocoder/rate (smaller, faster per language).

### Net

The end state genuinely simplifies adding a language at the **packaging/runtime
layer** (no JS G2P, no parity suite, `{metadata, model}` only) — exactly the
part that made German painful — while moving the linguistic work into the
**training pipeline**, where it only has to be done once and never shipped.

---

## Contributor experience — an "Add a language" guide (deliverable)

Making this **as easy as possible for contributors** is an explicit goal of the
work, not an afterthought. Each step must land with the docs and scaffolding a
new contributor needs, so "add a language" is a followable recipe rather than
tribal knowledge reverse-engineered from the German commit history.

**Deliverable: `docs/ADDING_A_LANGUAGE.md`** (written alongside the code, kept
current as the steps land), containing:

1. **A decision tree for the G2P** — rule-based (like German) vs. neural-ONNX
   (step 1 infra) vs. borrow an external phonemiser as a step-2 teacher — with
   the trade-offs and when to pick each.
2. **A concrete checklist** mirroring the real touch-points, each as a copy-able
   command:
   - register the language: symbols + tone offset + language ID
     (`src/dittli_tts/text/symbols.py`), then `npm run g2p:metadata`.
   - provide a G2P (rules JSON, or train + `scripts/export_g2p_onnx.py` for the
     neural path).
   - data + fine-tune (`scripts/finetune_*.py`, warm-start from an existing
     checkpoint) and export (`scripts/export_onnx`).
   - ship the pack and add a smoke test.
3. **A scaffolding command** — e.g. `scripts/new_language.py <code>` that
   stamps out the pack skeleton (`packages/tts-<code>/`), a metadata stub, and a
   placeholder smoke test, so a contributor starts from a working tree, not a
   blank page.
4. **What each step removes from the checklist**, so the guide shrinks as the
   work lands: step 1 drops "hand-port the G2P to JS + write a parity suite" for
   neural-G2P languages; step 2 drops the entire G2P column, leaving essentially
   "supply data + run the fine-tune."

**Success criterion:** a contributor can add a new language by following the
guide end-to-end without reading the engine internals, and the post-step-2 path
is "bring a dataset and a teacher G2P; run one training command."

## Validation gates

- **Step 1:** `scripts/export_g2p_onnx.py --verify` — ONNX greedy decode must
  match `g2p_predict.js` phoneme-for-phoneme on the test word list; measure the
  exported byte size; confirm the browser smoke app still synthesises.
- **CMU trim:** OOV rate <1 % on LJSpeech transcripts + Wikipedia leads.
- **Step 2/3:** distill loss → ~0 on held-out; blind MOS within 0.1 of current,
  n ≥ 20; parity of the char front-end Python↔JS (or move the front-end fully
  into the graph so there is no JS to keep in parity).
