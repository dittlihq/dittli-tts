# Runbook — G2P / model simplification

Living execution + verification guide for the roadmap in
[2026-06-01_PLAN_G2P_INTEGRATION.md](2026-06-01_PLAN_G2P_INTEGRATION.md). Tells
you (and future-us) **what to train, what to run, and how to verify** each
piece. Update it as steps land.

## Branch / PR state

| PR | Branch | Scope | Needs training? | State |
|----|--------|-------|-----------------|-------|
| #4 | `claude/review-plan-branch-xFV1C` | ORT `/wasm` fix + CI + Renovate | no | ready |
| #5 | `claude/fix-acronym-g2p` | acronym spell-out (off `develop`) | no | ready |
| #6 | `claude/g2p-model-integration` | plan + ONNX-bake PoC | no | ready |
| #7 | `claude/g2p-step1-onnx-runtime` | Step 1: neural G2P as ONNX | no | ready |

Stack order: #4 → #6 → #7. #5 is independent; only overlap is `g2p_en.js`
(see the bottom of this file).

---

## Verifying what already exists (no training)

All of these run today, in this sandbox or CI.

```bash
# JS unit + integration tests (host loop, engine, dittli-tts)
npm run test:js                      # expect 61 passing

# Python tests
uv sync --extra dev && uv run pytest # expect 89 passed, 2 skipped

# Lint
npm run lint:js                      # biome, exit 0
uvx ruff@0.15.12 check . && uvx ruff@0.15.12 format --check .
```

### G2P ONNX (Step 1) parity — two independent checks

```bash
# 1. Python: graphs vs the g2p_en library (also (re)bakes the FP16 assets)
uv run --extra dev --with onnx --with onnxruntime \
    python scripts/export_g2p_onnx.py --assets --fp16 --verify     # 10/10 match

# 2. JS: the *real committed graphs* through the actual createOnnxG2p host loop
npm i --no-save onnxruntime-node && node scripts/_g2p_onnx_parity.mjs  # 9/9 match
```

### Browser smoke (the one thing the sandbox can't do)

The browser path runs the same graphs + same host loop via `onnxruntime-web`,
but a real browser must confirm audible output before publishing:

```bash
cd examples/browser-vite
node copy-assets.mjs                  # copies pack assets incl. g2p_*.onnx into public/tts/
npm run dev                           # then synthesise EN + DE in the browser
```
Acceptance: EN sentence with an OOV word (e.g. "kubernetes") plays clearly; no
404 for `g2p_encoder.onnx` / `g2p_decoder_step.onnx` / `g2p_vocab.json`.

---

## What still needs TRAINING (Steps 2–3, not yet implemented)

These require a GPU run; warm-start from existing checkpoints
(`checkpoints/G.pth` EN, `checkpoints_de/G.pth` DE). Cloud guide:
[2026-04-28_TRAINING_DE.md](2026-04-28_TRAINING_DE.md) (Modal).

### Step 2 — unified character front-end (distillation)

**Goal:** model takes characters; delete the G2P assets entirely.

To build (does not exist yet):
1. Add a char→phoneme-latent front-end to `PhonemeEncoder`
   (`src/dittli_tts/models/synthesizer.py`) + an auxiliary tone head.
2. New distillation entry point (mirror `scripts/finetune_de.py` /
   `src/dittli_tts/training/modal.py`): teacher = current G2P+encoder, freeze
   flow/duration/vocoder, train only the front-end on (text → encoder
   activations).
3. Re-export; the pack's `g2p.prepare()` becomes a no-op.

**Verify:** distill loss → ~0 on a held-out batch; **blind MOS within 0.1** of
the current pipeline, n ≥ 20. Per-language: still needs a *teacher* G2P at
train time (rules / `espeak-ng` / `phonemizer`) — see "Impact on adding new
languages" in the plan.

### Step 3 — Vocos ISTFT vocoder + 22.05 kHz

**Goal:** model 4.6 → ~2.5 MB; ~5–8× faster inference.

To build:
1. `SAMPLING_RATE` 44100 → 22050 in `src/dittli_tts/utils/config.py`; re-derive
   `FILTER_LENGTH` / `HOP_LENGTH`; update both packs' `metadata.json` sample_rate.
2. Swap `WaveformDecoder` (`synthesizer.py`) for a Vocos head (ConvNeXt blocks +
   mag/phase + ISTFT).
3. Fine-tune ~50K steps, warm-started; resample training audio to 22.05 kHz once.

**Verify:** MOS within 0.1 of current at 22.05 kHz; re-run both parity checks +
the browser smoke after re-export.

Best done as **one fine-tune** combining Step 2 + Step 3.

### Cost / Modal budget

Grounded in the measured German run: **100K steps ≈ 10.2 h on an A10G
(~$1.10/hr) ≈ $11.20**; the `--max-steps 200` smoke was ~$0.10–0.30.

| Work | Per language | Both |
|------|-------------|------|
| Step 1 | $0 (no training) | $0 |
| Step 2+3 as one combined fine-tune (~50–100K steps, warm-started) | ~$6–12 | ~$12–24 |
| + realistic iteration (smoke + a re-run or two) | +50–100% | ~$25–40 |

**A ~$30 budget covers one clean combined fine-tune of both languages with a
thin margin** — a failed run or a tuning round can exceed it; budget $40–50 if
you expect to iterate. To stay inside $30:
- calibrate on **one** language first (~$11–15) before the second;
- always `--max-steps 200` smoke first (caught the German warm-start bug);
- persist a Modal data volume to skip the ~13 min cold-start preprocess;
- do Step 2 + Step 3 in a **single** fine-tune, not two passes.

---

## Open follow-ups (no training)

- **CMU frequency-trim (Lever 2/D):** subset `assets/en/cmudict.json` (5.3 MB)
  to the top ~30K words; neural ONNX handles the tail. Needs a word-frequency
  source (e.g. `wordfreq`). **Verify:** OOV rate <1 % on LJSpeech transcripts +
  Wikipedia leads; re-run `npm run test:js`. Intended as its own stacked PR.
- **Browser smoke** for #7 before publish (above).

## Merge-order note (#5 ↔ #7)

Both edit `packages/tts-en/src/g2p_en.js` (#5 adds `expandInitialisms` before
lowercasing; #7 makes the function async + swaps the OOV fallback to ONNX).
They compose — keep both. Recommended: merge #4, then #5, then rebase the
#6→#7 stack onto the updated `develop` and fold #5's one line into #7's async
function.
