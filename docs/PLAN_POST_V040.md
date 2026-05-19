# Post-v0.4.0 roadmap — runtime + G2P payload reduction

## Context

After v0.4.0 (clean asset layout, no `import.meta.url`, consumer ergonomics fixed),
the remaining payload is dominated by **runtime + asset bloat**, not by the TTS
model itself. The model is 4.6 MB FP16. The runtime around it is multiples of that.

This doc evaluates two structural changes that could shrink the cold-cache
footprint by ~5×, and proposes a phased rollout. It's an opinion paper to inform
sequencing — not a commitment to build.

The two levers:

1. The `onnxruntime-web` WASM runtime (13–26 MB).
2. The English G2P pipeline (9.4 MB: CMU dict + GRU seq2seq fallback).

---

## Lever 1: replace `onnxruntime-web`

### What we ship today

- TTS model (FP16 ONNX): **4.6 MB** per language
- `ort-wasm-simd-threaded.wasm`: **13 MB** (CPU/SIMD path — what we use)
- `ort-wasm-simd-threaded.jsep.wasm`: **26 MB** (WebGPU JSEP variant — unused)
- MJS shims: ~50 KB combined
- Total ORT footprint per consumer: **39 MB cold cache** if both variants ship;
  **13 MB** with only the CPU path.

The model uses Opset 14 with conventional ops: 231 Conv, 5 ConvTranspose, 67
MatMul, plus Gather/Expand/CumSum/Pad/MultiHeadAttention/LayerNorm. No custom
kernels. The normalizing flow's invertibility is baked into the export — at
inference time it's straight-line forward ops.

### Three replacement paths, in order of effort

**A. Trim what we already have (≤1 day).**
Stop shipping the JSEP variant. `copy-ort-wasm.js` currently copies both jsep
and non-jsep — we use the wasm execution provider only, so jsep is dead weight.
Removing two entries from `ASSETS` halves the runtime payload on the publish
side. No code change to core.

**B. Swap to `tract` via WASM (1–2 weeks).**
[`tract`](https://github.com/sonos/tract) is a Rust ONNX inference runtime
designed for embedded targets. Compiled to WASM with `wasm-pack`, it lands at
~1–2 MB for a comparable op set. Supports Opset 14 and all ops the model uses.
Effort: write a thin JS binding, port the `src/ort.js` interface, run parity
tests against ORT output on a held-out batch. Risk: 20–50 % slower than
ORT-WASM-SIMD (tract is single-threaded and less aggressively vectorised).
Mitigation: profile first; SIMD intrinsics in tract are improving.

**C. Hand-rolled C/Rust inference (4–8 weeks).**
Implement only the ops the model uses, against the specific topology. Could
land at 100–300 KB WASM. Building blocks: `ggml` (pure C, MIT, small WASM
build), `candle` (HF Rust, supports WASM), or a custom graph-codegen from the
ONNX file.

Breakdown:
- Op implementations (Conv1d/ConvTranspose1d/MatMul/MHA/LayerNorm/GLU/activations
  + flow reversal): ~3 weeks
- SIMD pass (wasm-simd128 intrinsics): ~1 week
- Threading via SharedArrayBuffer + COOP/COEP headers: ~1 week
- Test harness against ORT, perf tuning, bug fixes: ~2 weeks
- Plus: maintenance — every model architecture change requires a runtime update.

### Recommendation

**Do A immediately. Do B as a v0.6.0 candidate. Don't do C yet.**

The 26 → 13 MB win from A is free and unblocks the rest of the conversation.
The 13 → 1–2 MB win from B is ~10×, has a known unknown (perf), and is
contained — if tract is too slow we revert with no API change, since
`packages/tts-core/src/ort.js` is already the single touchpoint (this was the
load-bearing reason that abstraction exists; see PLAN_V040.md).

Option C is the right answer eventually, but only after the project has enough
traction to justify owning a perf/correctness lifecycle. ORT WASM is a known
quantity; a custom runtime is one critical bug away from a production fire.

**Side benefit of B/C:** killing the `[W:onnxruntime]` warning filter
monkey-patch in `src/ort.js:_installWarnFilter`. Library code patching
`console.warn` is something we apologise for in the README; a custom runtime
simply wouldn't emit those warnings.

---

## Lever 2: replace English G2P with a neural net (ideally unified)

### What we ship today (English only)

- `cmudict.json`: **5.1 MB** (133K word → phoneme lookup table)
- `g2p_model.json`: **4.3 MB** (GRU seq2seq, ~831K params, 256-dim hidden,
  base64-in-JSON)
- `g2p_en.js` + `g2p_predict.js`: ~10 KB code

**English G2P payload (9.4 MB) is bigger than the TTS model (4.6 MB).** This is
the most absurd line in the asset budget.

German is fine: its rule-based G2P is a tiny JSON of rules — the payload
problem is English-specific.

### The TTS model's input contract

The encoder embeds phoneme IDs via `nn.Embedding(n_vocab=280,
hidden_channels=192)`. It also embeds tones (4 levels for EN) and language IDs
separately. Tones are derived from ARPA stress markers in CMU output. **The
encoder does not see characters today.** Switching to character input is a
real architecture change.

### Four options

**A. Smaller drop-in neural G2P (1–2 weeks).**
Train a compact transformer or LSTM G2P on the entire CMU dict + a dictionary
like Wiktionary IPA. Target: ~1 MB at FP16. Drops CMU dict entirely; same input
contract to the TTS model. Quality risk: regression on in-vocab words that
today get a perfect CMU lookup. Mitigation: train with strong CMU-fit objective
(overfit on CMU on purpose; it's the ground truth for the common vocabulary).

- Asset win: 9.4 MB → ~1 MB
- TTS retraining: **none**
- Risk: medium (G2P quality is audible)
- Effort: 1–2 weeks training + listening tests

**B. Unified front-end via distillation (2–3 weeks).**
Prepend a small character→phoneme-embedding module to the existing
`PhonemeEncoder`:

```
chars → [char_emb + 1–2 transformer blocks] → phoneme-shaped tensor →
  existing encoder → ...
```

Train the prepended module to **reproduce the existing encoder's intermediate
activations** for matched (text, audio) pairs, using the current G2P+encoder as
the teacher. The rest of the model (flow, duration, vocoder) stays frozen.
Classic knowledge distillation: the teacher provides supervision, no audio
loss needed during this stage, training cost dominated by cheap forward passes.

- Asset win: 9.4 MB → ~0 MB (G2P becomes graph weights, +200–500 KB to the ONNX)
- TTS retraining: only the new front-end; main model frozen
- Risk: medium-low (distillation is well-understood; falls back to fine-tuning
  end-to-end if quality drops)
- Effort: 2–3 weeks (data prep, training loop, eval). Reuses existing training
  infra; distillation is a different loss head.
- Bonus: works for any language once you have a teacher G2P. Path to dropping
  the German rule engine too.

**C. Char-level end-to-end (3–4 weeks).**
Drop phonemes entirely at inference; train the model to pronounce characters
directly. Tones become a problem (currently derived from ARPA stress), so
either:
- Auxiliary tone-prediction head supervised by the current G2P at training time
- Drop tone conditioning (regression risk, especially for Chinese-derived
  phonemes already in the symbol set)

Most invasive option. Requires re-training the encoder embedding from scratch;
existing weights only partially carry over.

- Asset win: 9.4 MB → 0 MB
- TTS retraining: full encoder + downstream fine-tune
- Risk: high (tone modelling, alignment convergence)
- Effort: 3–4 weeks

**D. Trim CMU dict only (1 day).**
The full 133K-word CMU dict is overkill — most usage is the top ~30K. Ship the
30K most-frequent words and let the neural fallback handle the rest. Asset
win: 5.1 MB → ~1.5 MB CMU + still 4.3 MB G2P model. Not as clean as B/C but
lands the biggest single-day win.

### Recommendation

**D first (cheap win), B second (unified front-end via distillation).
Skip A and C.**

Reasoning:
- D is a half-day spike: subset CMU by frequency, measure OOV rate on a
  held-out corpus, ship. Buys 3–4 MB without touching the model.
- B is the cleanest long-term answer. The teacher–student setup means training
  cost is dominated by forward passes of the existing model on text, which is
  cheap. The Viterbi alignment infrastructure
  (`src/dittli_tts/alignment/core.py`) supplies frame-accurate phoneme
  positions if we ever want character→phoneme-position supervision instead of
  pure activation distillation.
- A is fine but lacks B's payoff (B eliminates the G2P entirely; A still ships
  one).
- C is the academic darling but the tone-modelling regression risk is real,
  and we'd lose the speaker/language-conditional structure the encoder relies
  on.

### Why B is more attractive than it looks

1. **The current `Engine.load()` already calls
   `pack.g2p.prepare({assetBase, signal, onProgress})`**
   (`packages/tts-core/src/engine.js:87–89`). If B replaces G2P with weights
   baked into the ONNX, the `prepare` hook becomes a no-op and the pack's
   `assets` shrink to just `{metadata, model}`. The v0.4.0 API survives
   unchanged. Back-end-only change.
2. **The neural G2P (`g2p_predict.js`) currently does GRU inference in pure
   JS.** That's a measurable per-utterance cost. Moving it into the ONNX graph
   runs on the same SIMD-vectorised WASM path as the rest of the model —
   likely a small inference-time win in addition to the asset win.
3. **The German pack gets the same treatment for free.** Today its G2P is JS
   rules. Tomorrow it's a tiny char-encoder front-end. Same contract, less
   per-pack JS, more uniformity across language packs.

---

## Combined roadmap

Ordered by ROI:

1. **v0.4.1 (a few days):** drop JSEP wasm (Lever 1 / A) +
   frequency-trim CMU dict (Lever 2 / D). 13 MB + 4 MB = ~17 MB of cold-cache
   payload removed, zero code-architecture risk.
2. **v0.5.0 (2–3 weeks):** unified character front-end via distillation
   (Lever 2 / B). The model graph eats the G2P. English asset drops to ~5 MB
   total.
3. **v0.6.0 (1–2 weeks):** evaluate `tract` as a runtime swap (Lever 1 / B).
   If perf parity holds, 13 MB → 1–2 MB runtime. If not, defer a custom runtime
   to a separate workstream when the time is right.

After all three, total cold-cache payload per language drops from ~40 MB today
to **~6–8 MB**. The "tiny TTS" framing becomes accurate at the runtime level
too, not just at the model level.

---

## Critical files referenced

- `packages/tts-core/src/ort.js` — single ORT touchpoint; the abstraction that
  makes Lever 1 swap-friendly
- `packages/tts-core/scripts/copy-ort-wasm.js` — drop jsep entries here for
  Lever 1 / A
- `packages/tts-core/src/engine.js:59-89` — `Engine.load()`; the
  `pack.g2p.prepare(...)` hook becomes optional when G2P moves into the graph
- `src/dittli_tts/models/synthesizer.py:234` — `PhonemeEncoder`; the host for
  the prepended character module in Lever 2 / B
- `src/dittli_tts/text/symbols.py:124-165` — English phoneme symbol table;
  reference for character-vocab mapping
- `src/dittli_tts/training/trainer.py:172-240` — training step; distillation
  loss head plugs in here
- `src/dittli_tts/alignment/core.py:14-46` — Viterbi alignment; supplies
  phoneme-position supervision if needed
- `packages/tts-en/src/g2p_predict.js` — JS-side GRU that gets replaced by
  graph weights in Lever 2 / B
- `packages/tts-en/assets/en/cmudict.json` — Lever 2 / D subsets this

## How to validate before committing to each step

- **Lever 1 / A (drop JSEP):** confirm the smoke app in
  `examples/browser-vite/` still plays end-to-end with only
  `ort-wasm-simd-threaded.{mjs,wasm}` copied to `public/tts/ort/`. Measure
  first-load network bytes in DevTools.
- **Lever 1 / B (tract):** parity test — generate 100 utterances with ORT,
  100 with tract from the same model + text, compare sample-wise L∞ and a
  perceptual metric (MOS or PESQ). Measure p50/p99 synthesis latency.
- **Lever 2 / D (CMU trim):** measure OOV rate on a held-out English corpus
  (LJSpeech transcripts, Wikipedia leads). Target <1 % OOV with 30K words.
- **Lever 2 / B (unified front-end):** distillation loss should track to
  near-zero on a held-out batch. Final eval: MOS listening test against the
  current pipeline, n ≥ 20 utterances, blind comparison.
