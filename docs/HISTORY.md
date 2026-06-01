# Project history

A chronological, summarised record of the major steps the project has gone
through. Each entry links to the deeper doc where one exists. For a
release-oriented view see [`/CHANGELOG.md`](../CHANGELOG.md); this file is the
narrative behind it (decisions, dead ends, gotchas).

Newest first.

---

## Summary timeline

| Date | Step | Outcome |
|------|------|---------|
| 2026-06-01 | ORT jsep import fix + docs reorg | Runtime fetch 25 MB → 12.4 MB via one-line import; FP16 kept |
| 2026-05-27 | rten runtime spike (branch only) | rten ≈1.6 MB but lost FP16, slower, brittle — kept as a spike, not merged |
| 2026-05-19 | **v0.5.0** — clean-break browser API | `DittliTTS` class + `assetBase`; JSEP intent dropped; 58/58 vitest |
| 2026-05-18 | **v0.4.0** — asset layout + ergonomics | Clean assets, no `import.meta.url`, single-touchpoint `ort.js` |
| 2026-04-30 | npm workspace split + **v0.1.0/0.2.0** | `@dittli/tts-{core,en,de}`; 0.2.0 last public on npm |
| 2026-04-29 | German training run (Thorsten Voice) | Trained, exported, browser-verified German model |
| 2026-04-28 | German implementation (Phases 1–5) | Symbols, G2P parity, training infra, ONNX export |
| 2026-04 | Rename + restructure (TinyTTS → Dittli) | `src/` layout, uv, linting, pytest suite |

---

## 2026-06-01 — ORT jsep import fix + docs reorganisation

**Problem.** `onnxruntime-web` was suspected unviable because of a "huge jsep
binary required at runtime." Root-caused on `develop`: `src/ort.js` imported the
**default** package entry (`import * as ort from "onnxruntime-web"`), whose
loader hard-references `ort-wasm-simd-threaded.jsep.wasm` (**25 MB**, the
WebGPU/JSEP build) and fetches it at runtime — even though we only use the CPU
`wasm` provider and only ship the non-jsep binary. The shipped wasm didn't even
match what the loader requested.

**Investigation (on a fresh install, onnxruntime-web `^1.19.0` → resolved
1.26.0):** grepped the dist bundles for the wasm filename each entry fetches.

| Import | JS bundle | `.wasm` fetched at runtime |
|--------|-----------|----------------------------|
| `"onnxruntime-web"` (was) | `ort.bundle.min.mjs` (0.39 MB) | `ort-wasm-simd-threaded.jsep.wasm` — **25.0 MB** |
| `"onnxruntime-web/wasm"` (now) | `ort.wasm.bundle.min.mjs` (0.07 MB) | `ort-wasm-simd-threaded.wasm` — **12.4 MB** |

**Fix.** One line: `src/ort.js` now imports `onnxruntime-web/wasm`. Bumped the
dependency to `^1.26.0` (newest). Runtime fetch halves (25 → 12.4 MB), FP16
model stays 4.6 MB, no toolchain. This made the rten migration and a reduced
custom ORT build *optional* rather than necessary — see
[2026-05-27_PLAN_POST_RTEN.md](2026-05-27_PLAN_POST_RTEN.md) (updated header).

Also: renamed the docs in this folder to carry their creation date as a prefix,
renamed `SESSION_NOTES.md` → this file, and added a root `CHANGELOG.md`.

## 2026-05-27 — rten runtime spike (branch `size-improvements`, not merged)

Migrated the inference runtime from `onnxruntime-web` to
[`rten`](https://github.com/robertknight/rten) (Rust ONNX runtime compiled to
WASM, vendored as `packages/tts-runtime/` + a patched `rten-simd-patched`).
Runtime WASM dropped to ~1.6 MB, **but**: rten has no fp16 kernels so the model
bloated 4.6 MB FP16 → 6.9 MB FP32 per language, inference was slower
(single-threaded SIMD), and the export needed graph workarounds (int64→i32
casts, single-output assumption). Written up in
[2026-05-27_PLAN_POST_RTEN.md](2026-05-27_PLAN_POST_RTEN.md). Superseded by the
2026-06-01 jsep finding; kept as a recorded spike, not merged.

## 2026-05-19 — v0.5.0: clean-break browser API reshape

Replaced the v0.2.0 integration workaround with a ~10-line consumer setup:
`new DittliTTS({ language, assetBase })` + `tts.play(text)`. The class owns the
`AudioContext` and is AbortSignal-aware; `synthesize()` exposes pure inference.
Removed the old `register*` / `modelUrl` / WAV-bytes `speak()` surface (no shim).
Assets relocated to `assets/<lang>/`. `src/ort.js` established as the single
`onnxruntime-web` touchpoint; the copy script's *intent* was to drop the JSEP
variant (26 → 13 MB) — though the default-entry import kept pulling jsep until
the 2026-06-01 fix. Language packs became `peerDependency` of core; `.d.ts`
shipped from every package. 58/58 vitest (happy-dom). Tagged `v0.5.0`. Roadmap
captured in [2026-05-19_PLAN_POST_V040.md](2026-05-19_PLAN_POST_V040.md).

## 2026-05-18 — v0.4.0: asset layout + consumer ergonomics

Clean asset layout, removal of module-level `new URL(..., import.meta.url)`
(which had blocked English under Vite dep pre-bundling), and the consumer-side
ergonomics that v0.5.0 built on. Design rationale in
[2026-05-18_PLAN_V040.md](2026-05-18_PLAN_V040.md).

## 2026-04-30 — npm workspace split (v0.1.0 → v0.2.0)

Split the single package into a workspace: `@dittli/tts-core` (ONNX engine +
CLI), `@dittli/tts-en` (English G2P, CMU dict, neural fallback), `@dittli/tts-de`
(German rules). Retargeted the build from Node to the browser. Published `0.1.0`,
then `0.2.0` (the last public release before the v0.5.0 clean break; `0.3.x`/
`0.4.x` were skipped on npm).

## 2026-04-28 — German implementation (Phases 1–5)

Added German alongside the English-only base model. Summary (full detail in
[2026-04-28_PROGRESS.md](2026-04-28_PROGRESS.md) and
[2026-04-28_PLAN_DE.md](2026-04-28_PLAN_DE.md)):

- **Phase 1** — symbol-table extension (`de_symbols`, `num_de_tones=1`) +
  checkpoint embedding remapper; 219-symbol English snapshot saved.
- **Phase 2** — Python + JS G2P, parity-tested 805/805.
- **Phase 3** — training infra: `forward()`, MPD discriminator, losses, audio
  module, Thorsten dataset + preprocess, single-GPU AMP trainer, fine-tune.
- **Phase 4** — browser multi-language support (metadata-driven G2P dispatch).
- **Phase 5** — portable, sidecar-aware ONNX export.

## 2026-04 — Rename + restructure (TinyTTS → Dittli TTS)

Renamed `tiny_tts/` → `dittli_tts/`, reorganised into a `src/` layout with a
training/inference split, moved dev tools into `tools/` and historical docs into
`docs/`, dropped `requirements.txt` for `pyproject.toml` + `uv`, added Biome
(JS) and Ruff (Python) linting, and added a full pytest suite (unit/parity/
integration/slow). Replaced a mismatched checkpoint (`hidden_channels=80`) with
the correct `hidden_channels=32` checkpoint. Added `AGENTS.md`.

---

# Appendix: German training session log — 2026-04-29

Chronological session log: from a partially-implemented German pipeline
to a trained, exported, browser-verified German TTS model. Captured here
as an archival record; for current repo state see
[2026-04-28_PROGRESS.md](2026-04-28_PROGRESS.md),
for the original plan see [2026-04-28_PLAN_DE.md](2026-04-28_PLAN_DE.md), and for the cloud
training guide see [2026-04-28_TRAINING_DE.md](2026-04-28_TRAINING_DE.md).

## Goal

Take a fork of DittliTTS (1.6 M-param VITS-derived browser TTS, English
only, training code stripped) where Phases 1–5 of the German support
plan had been implemented but not exercised, and:

1. Validate the pipeline with cheap pre-training checks.
2. Train a German checkpoint on Thorsten Voice (~23 h CC0 single-speaker
   dataset) on a cloud GPU.
3. Export to ONNX and verify the npm-package browser path works.

## Repo & branch

- **Repo:** `github.com/brio1009/dittli-tts`
- **Branch:** `claude/german-implementation-progress-15W8v` (off `develop`)

## What happened, in order

### 1. Pre-training validation

Walked through the four cheap gates before any GPU spend:

| # | Check | Command |
|---|---|---|
| 1 | G2P parity (Python ↔ JS) | `python scripts/test_g2p_parity.py` → 805/805 |
| 2 | Symbol-table size | `python -c "from dittli_tts.text.symbols import symbols; print(len(symbols))"` → 220 |
| 3 | Dataset preprocess | `python -m dittli_tts.data.preprocess ...` |
| 4 | CPU smoke test | `python scripts/smoke_de.py ...` |

### 2. Setup script broken (404)

`scripts/setup_de_data.sh` pointed at an OpenSLR URL that no longer
existed. Two issues at once:

- The 2022.10 release moved to Zenodo as a `.zip` (not `.tgz`).
- The 2022.10 archive ships `metadata_train.csv` / `_dev.csv` / `_test.csv`
  rather than a single `metadata.csv`, and includes `__MACOSX/` resource
  forks.

**Fix:** rewrote the script to:
- Fetch from Zenodo (`ThorstenVoice-Dataset_2022.10.zip`, 1.4 GB) with
  the OpenSLR `thorsten-de_v02.tgz` as fallback.
- Detect `.zip` vs `.tgz` and use the right extractor.
- Flatten any single nested top-level dir.
- Strip `__MACOSX/`.
- Concatenate the three split metadata files into a single `metadata.csv`.
- Recover from half-extracted state without re-downloading.

### 3. `torchaudio.load()` failed (torchcodec)

In torchaudio ≥ 2.6, `torchaudio.load()` delegates to the separate
`torchcodec` package, which isn't installed (and pulls FFmpeg).

**Fix:** in `dittli_tts/audio.py`, replaced `torchaudio.load()` with
`soundfile.read()` (already in `requirements.txt`). Also fixed
`_mel_basis()` to use the imported `AF` alias instead of bare
`torchaudio.functional`.

### 4. CPU smoke ran cleanly

`scripts/smoke_de.py` produced finite losses, asserted all components,
backward pass succeeded. Pipeline validated.

### 5. Cloud GPU choice

User had ~$30/month free Modal credits and didn't want to babysit a
notebook. Compared options:

| Option | Cost | Babysitting? |
|---|---|---|
| Kaggle interactive | Free | Tab must stay open, 12 h cap |
| Kaggle Save & Run All | Free | No, but 12 h commit cap |
| Modal | ~$10–20 | None (`--detach`) |
| Vast.ai with `nohup` | ~$15–25 | None (SSH) |

Picked **Modal** for ergonomics + free credit coverage.

### 6. Created `modal_train.py`

Modal entrypoint with:
- Persistent volume `dittli-de` for checkpoints (~50 MB long-term).
- Ephemeral `data/thorsten` symlinked to `/tmp/thorsten` so the ~38 GB
  spec cache doesn't bloat the volume.
- Resume-from-volume: picks highest numeric `G_<step>.pth` if present,
  else falls back to English warm-start.
- 12 h timeout with `try/finally` `volume.commit()` so partial progress
  survives killed runs.

### 7. Three iterations to get the smoke run green

| Failure | Fix |
|---|---|
| `curl: command not found` | Added `curl` to `apt_install` list. |
| `metadata.csv not found after extract` | `find` won't follow symlinked starting points by default; rewrote `flatten_dataset()` to use shell glob. |
| `ValueError: invalid literal for int() with base 10: 'final'` | Resume lambda crashed on `G_final.pth` left by smoke; filtered to numeric step names only. |

### 8. Found a real warm-start bug

Smoke log printed `[trainer] generator: loaded 996 tensors, skipped 1`.
Investigated: the symbol snapshot `checkpoints/symbols_v1_en.txt` is
needed for the embedding remap, but the `add_local_dir` `ignore` list in
`modal_train.py` excluded the whole `checkpoints/` directory. Result:
trainer fell back to its "no remap" branch and randomly initialized the
phoneme embedding.

**Fix:** changed ignore from `checkpoints/**` → `checkpoints/*.pth`.
Verified next run logged `loaded 997, skipped 0`.

### 9. Full training run

```bash
modal run --detach modal_train.py
```

- 100 000 steps in 36 631 s (~10.2 h) on A10G at ~$1.10/hr.
- **Cost: ~$11.20.**
- Final losses: `mel=23.9, kl=1.7, dur=1.9, d=1.5, adv=3.9, fm=8.7`.
  Late-stage discriminator-pulls-ahead pattern, mel converged ~step
  1500 onward and held flat. No NaNs throughout.

### 10. ONNX export

```bash
pip install onnxscript onnxruntime
python export_onnx.py --checkpoint G_de.pth --lang DE --out models/dittli-de.onnx
```

Initial export failed because newer torch versions default `torch.onnx.export`
to the dynamo path, which couldn't lower this VITS graph (sympy shape
expressions like `Sym(s11**2 + s11*(s11-1))` and inline `torch.randn`
in the SDP).

**Fix:** added `dynamo=False` to force the legacy TorchScript tracer.
FP32 export succeeded (~6 MB), then FP16 conversion via `onnxruntime`
produced `models/dittli-de_fp16.onnx` (~3 MB).

### 11. End-to-end browser path verified

```bash
node npm-package/bin/cli.js "Guten Morgen, wie geht es dir?" \
    --model models/dittli-de.onnx -o de.wav
```

Produced intelligible German. Done.

## Final artifacts

- `G_de.pth` — local checkpoint (`*.pth` is gitignored; pull from Modal
  volume `dittli-de` at `checkpoints_de/G_final.pth`).
- `models/dittli-de.onnx` (FP32, ~6 MB).
- `models/dittli-de_fp16.onnx` (FP16, ~3 MB).
- `models/dittli-de.json` — sidecar (committed).

## Files changed this session

**Modified:**
- `scripts/setup_de_data.sh` — Zenodo URL, format detection, glob-based
  flatten, metadata concatenation, idempotent recovery.
- `dittli_tts/audio.py` — soundfile-based loader.
- `export_onnx.py` — `dynamo=False`.

**Created:**
- `modal_train.py` — Modal entrypoint with resume-aware checkpoint volume.

## Lessons / gotchas worth remembering

1. **`find` doesn't follow command-line symlinks by default.** Use shell
   glob (`for d in $DIR/*/`) for traversal that needs to descend into a
   symlink starting point. Or pass `-L` to find.
2. **Modal `add_local_dir` ignore patterns** — easy to over-exclude.
   `checkpoints/**` looks reasonable but kills `checkpoints/*.txt`
   alongside `*.pth`. Be granular.
3. **`torch.onnx.export` default flipped to dynamo** in recent versions
   (~2.6+). Pass `dynamo=False` if your script was written for the
   legacy tracer.
4. **`torchaudio.load` now requires torchcodec** in 2.6+. For wav-only
   pipelines, `soundfile` is a much lighter dep.
5. **VITS late-training pattern is normal:** discriminator pulls ahead,
   `fm` and `adv` rise, `mel` plateaus. Audio quality lives in `mel`
   and the actual wav output, not the headline `g` total.
6. **Resume-aware scripts must reject sentinel filenames.** `G_final.pth`
   from a 200-step smoke poisoned a real-run resume. Filter to numeric
   step names; a complete real run has both `G_<final_step>.pth` and
   `G_final.pth` with identical weights, so dropping `G_final` is safe.

## Discussed but not implemented

- **Persistent Modal data volume** to skip ~13 min cold-start
  download + preprocess on reruns. Only worth it for multi-run iteration.
- **Per-language npm split** (`dittli-tts-en`, `dittli-tts-de`) so users
  only ship the G2P assets they need.
- **Angular integration pattern** — lazy `import()` of the package
  inside a feature module, `<link rel="prefetch">` for the ONNX so
  first-paint is unaffected and first-use latency is near-zero.
- **v2 G2P polish** — glottal stop `ʔ`, expanded loanword exception
  dict (~1 % naturalness gain).
