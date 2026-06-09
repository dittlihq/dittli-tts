# Post-rten: a better path to fast, small, in-browser TTS

> **Update 2026-06-01 — the jsep pain was an import bug, not an ORT
> limitation. Phase 1 below is largely moot; the runtime swap is optional.**
>
> This plan's premise is that `onnxruntime-web` is unviable at its default
> ~13 MB and that escaping it (via rten, or a reduced custom build) is the
> path forward. Investigation on `develop` showed the real problem was
> narrower: `src/ort.js` imported the **default** package entry
> (`onnxruntime-web`), whose loader hard-references
> `ort-wasm-simd-threaded.jsep.wasm` (**25 MB**, the WebGPU/JSEP build) and
> fetches it at runtime — even though we only use the CPU `wasm` provider and
> only *ship* the non-jsep binary. That mismatch is the "huge jsep required
> at runtime" we hit.
>
> The fix is one line: import the WASM-only subpath
> `onnxruntime-web/wasm`, whose loader references the non-jsep
> `ort-wasm-simd-threaded.wasm` (**12.4 MB**) — the file `copy-ort-wasm.js`
> already ships — and pulls a smaller JS bundle (0.39 MB → 0.07 MB).
> Verified against onnxruntime-web 1.26.0 by grepping the dist loaders and
> resolving the subpath. **FP16 model stays at 4.6 MB; no Rust crate, no
> emscripten toolchain, no rten.** Applied in this branch.
>
> What this means for the rest of the doc:
> - **Phase 1 (runtime swap to a reduced ort-web build): demoted to
>   optional.** It's still a legitimate *size* optimization if 12.4 MB CPU
>   wasm proves too heavy in practice (12.4 → ~2–4 MB, unverified), but it
>   now buys size only, not the jsep fix, and it costs an ORT-from-source
>   build to own in CI. Gate it on a real measurement before committing.
> - **rten is a sledgehammer** for a problem a one-line import solved, and
>   it regresses FP16 (model +2.3 MB), speed, and brittleness (int64→i32 +
>   single-output hacks). Keep the `size-improvements` branch only as a
>   recorded spike.
> - **Phases 2 and 3 (Vocos ISTFT vocoder, 22.05 kHz output, INT8 PTQ)
>   stand unchanged** — they are model-side wins independent of the runtime
>   question and remain the strongest content here.
>
> The original text is preserved below for the record.

---

## Context

After PLAN_POST_V040, we migrated from `onnxruntime-web` (13 MB WASM) to
`rten` (1.6 MB WASM). The migration shrank the runtime, but it traded one set
of problems for another:

- **rten has no fp16 kernels**, so the exported model bloated from 4.6 MB FP16
  → **6.9 MB FP32** per language. We *gave back* 2.3 MB of the model budget
  for every language we ship.
- **Inference is noticeably slow.** With FP32 throughout (double the memory
  bandwidth), the runtime is leaving a lot on the table. (Note: rten's
  rayon path doesn't auto-enable browser threads from a plain `wasm-pack`
  build, so today we're effectively single-threaded SIMD anyway.)
- **"Bunch of other workarounds"** — model-graph tweaks to fit rten's op
  coverage, which makes the export pipeline brittle and ties future model
  changes to runtime support.

We also learned, while exploring the codebase, that **the
`WaveformDecoder` (HiFi-GAN-style vocoder) dominates inference time**:
5 ConvTranspose1d stages with upsampling rates `[8,8,2,2,2]` (512× total)
plus 30 dilated Conv1d layers in ResBlocks — order-of-magnitude
2–5 GFLOPs per second of audio, roughly 60-80% of total inference cost.
Swapping the runtime is necessary but not sufficient.

A third observation: **the model outputs 44.1 kHz** (see
[src/dittli_tts/utils/config.py:2](../src/dittli_tts/utils/config.py#L2)),
but LJSpeech (our EN dataset) is natively **22.05 kHz**. We're upsampling
beyond the source data's actual frequency content. Halving the sample rate
to 22.05 kHz (the TTS standard) is quality-neutral, drops the last
upsample stage, and halves the vocoder's per-second compute. The user is
explicitly fine with a lower sample rate.

The three levers that actually matter:

1. A runtime that supports **fp16 + int8 + SIMD** out of the box (and
   crucially: doesn't require COOP/COEP headers — static hosting must
   stay frictionless).
2. **Replacing the HiFi-GAN vocoder** with an ISTFT-based head (Vocos-style)
   — same input contract, a fraction of the FLOPs.
3. **Halving the output sample rate to 22.05 kHz** — cuts the upsample
   stack and the audio buffer in half, for free.

This plan recommends doing all three, in two phases, with a sensible
rollback at each step.

---

## What we missed in PLAN_POST_V040

PLAN_POST_V040 rejected `onnxruntime-web` because the default build is 13 MB.
**It never considered the reduced-operator-kernel build.** Microsoft ships an
official toolchain (`tools/python/create_reduced_build_config.py` +
`build.py --include_ops_by_config`) that produces a WASM binary containing
only the ops our specific model uses. Realistic target: **1–2 MB**.

This is the runtime every other in-browser TTS project uses in 2026:

| Project   | Model size | Runtime           | Notes                          |
|-----------|-----------|-------------------|--------------------------------|
| Piper     | 20–60 MB  | ort-web (reduced) | de-facto standard for VITS     |
| Kokoro    | 82 MB     | ort-web + WebGPU  | WebGPU EP for 3× speedup       |
| KittenTTS | 15–25 MB  | ort-web + INT8    | Quantized PTQ                  |

We were swimming against the current. The right move is to swim with it —
ort-web reduced — and put our originality budget into the model itself
(vocoder + quantization).

---

## Execution scope

**This plan executes Phase 1 only.** Phases 2 and 3 are captured below as
*future work* — we'll re-evaluate after Phase 1 ships and we have real
telemetry on whether ort-web alone is fast enough.

**Default: no COOP/COEP headers required.** We ship the **SIMD-only,
non-threaded** ort-web build (`ort-wasm-simd.wasm`) as the primary
artefact. SIMD does not need cross-origin isolation; only
`SharedArrayBuffer`-based threading does. This keeps the library drop-in
for any static host (GitHub Pages, plain CDN, embedded widgets) — exactly
the friction-free deploy story v0.4.0 set out to preserve.

**Escape hatch: threaded build available behind a flag.** If Phase 1
benchmarks show single-threaded SIMD inference is unacceptably slow, we
will *also* ship `ort-wasm-simd-threaded.wasm` as an opt-in alongside the
default. Consumers who can serve their site with COOP/COEP headers
(`Cross-Origin-Opener-Policy: same-origin`,
`Cross-Origin-Embedder-Policy: require-corp`) point `wasmBase` at the
threaded directory and get the speedup; everyone else gets the
single-threaded SIMD path with no extra setup. We make this decision
based on the verification benchmarks, not upfront.

WebGPU is also **out of scope** for now. Keep the runtime WASM-only (one
code path, no Safari split). We can revisit if neither single-threaded SIMD
nor opt-in threading is enough.

---

## Phase 1 — Runtime swap to `onnxruntime-web` reduced build (~1 week)

**Goal:** match rten's WASM size, get FP16 back (model 6.9 → 4.6 MB),
regain threading + SIMD via a mature runtime, and drop the
rten-specific export workarounds.

Concrete steps:

1. Re-export the model with FP16 conversion enabled (already supported in
   [src/dittli_tts/inference/export.py](../src/dittli_tts/inference/export.py)
   lines 158–176). No graph changes — drop the rten-specific
   workarounds the migration introduced.
2. Run Microsoft's reduced-build pipeline against the FP16 ONNX:
   - Extract the op list with `create_reduced_build_config.py`.
   - Build the **non-threaded SIMD** variant with `--include_ops_by_config`
     and `--enable_wasm_simd` (no `--enable_wasm_threads` for the default).
   - Expected output: ~1.5 MB `ort-wasm-simd.wasm` (down from 13 MB
     default, comparable to today's rten 1.6 MB). No COOP/COEP needed.
   - Optionally also build the threaded variant
     (`ort-wasm-simd-threaded.wasm`) as a sibling artefact. Ship both;
     consumers pick by pointing `wasmBase` at the appropriate folder.
     Decision on whether to ship the threaded variant is gated on the
     verification benchmarks below.
3. Replace [packages/tts-core/src/runtime.js](../packages/tts-core/src/runtime.js)
   with a thin ort-web wrapper. The
   [packages/tts-core/src/runtime-worker.js](../packages/tts-core/src/runtime-worker.js)
   pattern stays (off-main-thread inference is still the right shape); just
   the inference call inside changes. Tensor wire format
   (`{type, data, shape}`) stays the same so
   [packages/tts-core/src/engine.js](../packages/tts-core/src/engine.js)
   doesn't change.
4. Delete `packages/tts-runtime/` (Rust crate) and
   `packages/rten-simd-patched/` (the patched dep) from the workspace.
5. Update [packages/tts-core/scripts/copy-runtime-wasm.js](../packages/tts-core/scripts/copy-runtime-wasm.js)
   to copy the reduced-build ort-web artefacts (default:
   `ort-wasm-simd.{mjs,wasm}`; optional: the threaded sibling) instead of
   the rten outputs.
6. Hard-code `executionProviders: ['wasm']`. No WebGPU code path.
7. Document in the README that the default build needs **no special
   headers**, and that consumers who want threading set up COOP/COEP and
   point `wasmBase` at the threaded variant.

**Per-language asset delta from Phase 1:**

| Asset             | Today (rten) | After Phase 1 (ort-web reduced) |
|-------------------|--------------|----------------------------------|
| Runtime WASM      | 1.6 MB       | ~1.5 MB                          |
| Model             | 6.9 MB FP32  | 4.6 MB FP16                      |
| **Per-language**  | **8.5 MB**   | **6.1 MB**                       |

Inference speed: at parity or faster than rten today, even on the
single-threaded SIMD path (ort-web has more mature SIMD kernels than rten's
patched portable-simd path). If we ship the threaded sibling, opt-in
consumers get an additional ~2-4× on top.

**Risk:** low. ort-web is mature, the model exports cleanly to opset 14,
and the [packages/tts-core/src/runtime.js](../packages/tts-core/src/runtime.js)
abstraction is already the single touchpoint.

**Rollback:** keep the rten branch alive for 1 release if regressions appear.

---

## Future work (not in this commit)

### Phase 2 — Vocos ISTFT vocoder + 22.05 kHz output (~2 weeks)

**Goal:** kill the inference bottleneck and halve the output rate at the
same time, since we're doing a fine-tune anyway. The WaveformDecoder is
60-80% of inference cost; Vocos collapses it into a few transformer blocks
+ a single ISTFT call.

What Vocos is:
- Predict magnitude + phase spectra from the mel-conditioned latent.
- Apply ISTFT (one operation, browser-native via WebAudio or a small WASM
  shim) to reconstruct the waveform.
- No upsampling Conv stack at all.
- Published result: comparable MOS to HiFi-GAN, 13× faster on CPU.

Concrete steps:

1. Drop `SAMPLING_RATE` from 44100 → 22050 in
   [src/dittli_tts/utils/config.py:2](../src/dittli_tts/utils/config.py#L2).
   Re-derive `FILTER_LENGTH` / `HOP_LENGTH` for the new rate. Update both
   pack metadata files to `sample_rate: 22050`.
2. Swap the
   [`WaveformDecoder`](../src/dittli_tts/models/synthesizer.py) (line 394–461)
   for a Vocos-style head:
   - Backbone: 8 ConvNeXt-style 1-D blocks (already a graph-friendly op set).
   - Heads: two linear projections → mag + phase spectra.
   - ISTFT: implement either as a Conv-based op (graph-friendly) or expose
     mag/phase as the model output and call ISTFT in JS (native via
     `OfflineAudioContext` or a tiny FFT helper).
3. Initialize the **encoder + flow + duration predictor from the current
   English/German checkpoints** (frozen for the first epoch). Only the new
   vocoder head trains from scratch. The 22.05 kHz target means we
   resample the existing 44.1 kHz training audio down once before training
   — no other dataset changes.
4. Fine-tune for ~50K steps on the existing
   [LJSpeech (EN)](../src/dittli_tts/inference/engine.py) and
   [Thorsten Voice (DE)](../src/dittli_tts/data/dataset.py) datasets.
   Existing training infra
   ([src/dittli_tts/training/trainer.py](../src/dittli_tts/training/trainer.py)
   + [src/dittli_tts/training/modal.py](../src/dittli_tts/training/modal.py))
   supports warm-start from `init_g_ckpt`. ~1-2 days on a single A10G.
5. Re-export, run reduced-build pipeline against the updated op set.

**Per-language asset delta from Phase 2:**

| Asset            | After Phase 1 | After Phase 2 |
|------------------|----------------|----------------|
| Runtime WASM     | ~1.5 MB        | ~1.5 MB        |
| Model            | 4.6 MB FP16    | ~2.5 MB FP16   |
| **Per-language** | **6.1 MB**     | **~4.0 MB**    |

Inference speed: vocoder cost drops by ~10× from Vocos, additional ~2×
from halving the sample rate. End-to-end should land at **5–8× faster than
today's rten** on WASM. Audio quality target: MOS within 0.1 of current
model on a 20-utterance blind test (22.05 kHz is the standard TTS rate and
should not cause audible regression for speech).

**Risk:** medium. Vocoder quality is audible, but Vocos is well-validated
(2023 paper, multiple reproductions). The sample-rate drop is essentially
free since LJSpeech is natively 22.05 kHz. Falls back to keeping
HiFi-GAN+44.1 kHz if MOS regresses unacceptably — we just stay with Phase 1.

### Phase 3 — Optional INT8 PTQ (~3 days)

Post-training quantization via ort-web's quantization toolchain. Targets:

- 2.5 MB FP16 model → ~1.3 MB INT8 model
- 1.5-2× speedup on CPUs with INT8 GEMM support
- Quality risk: small; ort-web's calibration on a few hundred utterances
  usually keeps MOS regression below 0.1.

Defer this until Phase 1+2 ship and we have telemetry on whether it's still
worth it. If model size is no longer the bottleneck after Phase 2 (~4 MB
total per language), INT8 may not be worth the calibration step.

---

## Per-language cold-cache footprint

| Stage                       | Per-language total | Note                   |
|-----------------------------|--------------------|------------------------|
| Today (rten + FP32, 44 kHz) | 8.5 MB             | Slow inference         |
| **After Phase 1 (target)**  | **6.1 MB**         | **At-parity or faster**|
| After Phase 2 (22.05 kHz)   | 4.0 MB             | ~5-8× faster than now  |
| After Phase 3 (optional)    | 2.8 MB             | Further 1.5-2× faster  |

Plus a one-time-shared ~1.5 MB runtime WASM across all languages, vs the
13 MB ORT default the original plan rejected.

---

## Why not the alternatives

- **GGML port (C++ via emscripten).** Smallest WASM theoretically, but no
  production browser-TTS deployments to point at. We'd be the canary. Higher
  effort, similar end size to ort-web reduced.
- **Candle (Rust HF).** 2026 momentum exists (Qwen3-TTS) but the ecosystem
  for tooling, debugging, and op coverage is still behind ort. Reasonable
  hedge in 12 months; not today.
- **WebNN API.** Origin-trial-only in Chrome 147–149 as of 2026-05.
  Not shippable.
- **Direct WebGPU compute shaders.** Excellent for the vocoder hotspot
  specifically, but requires WGSL per op and breaks the Safari/non-WebGPU
  story. Park it; revisit if Phase 1+2 still leave us slow.
- **Stay with rten + push fp16/threads upstream.** rten has no fp16 roadmap
  and patching rayon for browser threads is non-trivial. We'd own the
  perf/correctness lifecycle without any of ort-web's ecosystem benefits.

---

## Critical files

**Runtime swap (Phase 1):**
- [packages/tts-core/src/runtime.js](../packages/tts-core/src/runtime.js) — replace rten worker plumbing with ort-web session
- [packages/tts-core/src/runtime-worker.js](../packages/tts-core/src/runtime-worker.js) — keep the off-main-thread pattern; swap the inference call
- [packages/tts-core/scripts/copy-runtime-wasm.js](../packages/tts-core/scripts/copy-runtime-wasm.js) — copy the reduced-build ort-web artefacts instead
- [packages/tts-core/src/engine.js](../packages/tts-core/src/engine.js) — no changes; the abstraction holds
- [src/dittli_tts/inference/export.py](../src/dittli_tts/inference/export.py) — re-enable FP16 conversion
- Delete: `packages/tts-runtime/`, `packages/rten-simd-patched/`

**Vocoder swap + sample rate (Phase 2):**
- [src/dittli_tts/models/synthesizer.py:394-461](../src/dittli_tts/models/synthesizer.py#L394) — replace `WaveformDecoder` with Vocos head
- [src/dittli_tts/utils/config.py:2](../src/dittli_tts/utils/config.py#L2) — `SAMPLING_RATE = 22050`; update `FILTER_LENGTH` / `HOP_LENGTH` for new rate
- [packages/tts-en/assets/en/metadata.json](../packages/tts-en/assets/en/metadata.json) and [packages/tts-de/assets/de/metadata.json](../packages/tts-de/assets/de/metadata.json) — `sample_rate: 22050`
- [src/dittli_tts/training/trainer.py](../src/dittli_tts/training/trainer.py) — supports warm-start via `init_g_ckpt`; add ConvNeXt block to imports
- [src/dittli_tts/training/modal.py](../src/dittli_tts/training/modal.py) — fine-tune entry point; no changes expected

## Verification (Phase 1)

- [examples/browser-vite/](../examples/browser-vite/) smoke app produces
  audible audio for EN + DE **without** COOP/COEP headers set (this is
  the key acceptance criterion for the default build).
- DevTools first-load Network panel shows ≤2 MB ort-web WASM and 4.6 MB
  model.
- Parity test: synth 100 utterances on rten vs ort-web from the same
  checkpoint; compare sample-wise L∞ ≤ 1e-3 and run an informal listening
  test for confirmation.
- Bench: p50/p99 synthesis latency for a 5-second utterance on the
  **single-threaded SIMD** build. Target: p50 ≤ current rten p50 (parity
  or faster). **Decision gate:** if p50 is acceptable, ship single-threaded
  only and document COOP/COEP as unnecessary. If p50 is too slow, also ship
  the threaded sibling and document the COOP/COEP setup as an opt-in
  speedup.
- Run [tests/js/](../tests/js/) suite end-to-end against the new runtime.
- Confirm the package publishes cleanly:
  [packages/tts-core/scripts/copy-runtime-wasm.js](../packages/tts-core/scripts/copy-runtime-wasm.js)
  produces the ort-web artefacts and the example consumer can pick them up
  from its static asset tree.
