# German Implementation Progress

This file tracks progress so a fresh session can resume work without re-reading
every file. Refer to PLAN_DE.md for the detailed plan.

## Branch
- Working on `claude/german-implementation-progress-15W8v`
- Push target: same branch

## Status Snapshot

### Completed previously (committed in 3d684d9)
- [x] Phase 1 — Symbol table extension
  - `tiny_tts/text/symbols.py`: `de_symbols = ["ʏ", "̩", "yː"]`, `num_de_tones = 1`
  - DE entries in `language_id_map` (id=7) and `language_tone_start_map`
- [x] Phase 1 — Embedding remapper: `tiny_tts/utils/remap_checkpoint.py`
- [x] Saved old symbol list snapshot: `checkpoints/symbols_v1_en.txt` (219 entries)
- [x] Phase 2a — Python G2P
  - `tiny_tts/text/german.py` (rule scanner, exception dict, callbacks)
  - `tiny_tts/text/german_utils/abbreviations.py`
  - `tiny_tts/text/german_utils/number_norm.py`

### Completed in this session
- [x] Phase 2b — JS G2P port + rules JSON generator
  - `scripts/gen_de_rules.py` (uses sys.modules shim to avoid eager torch import)
  - `npm-package/g2p_de_rules.json` — 76 rules, 37 exceptions, 28 abbreviations
  - `npm-package/g2p_de.js` — context-sensitive callbacks reimplemented in JS
- [x] Phase 2c — Parity harness
  - `scripts/test_g2p_parity.py` + `scripts/_run_js_g2p.js`
  - `scripts/de_test_words.txt` (805 German words)
  - **Verified: 805/805 phoneme sequences match between Python and JS.**
- [x] Phase 3a — `VoiceSynthesizer.forward()` added in
  `tiny_tts/models/synthesizer.py`. Uses MAS via the existing
  `alignment.viterbi_decode`, expanded prior, KL latents, segment slicing.
- [x] Phase 3b — `tiny_tts/models/discriminator.py`
  (HiFi-GAN MPD with periods 2/3/5/7/11 + a single MSD branch).
- [x] Phase 3c — `tiny_tts/losses.py` (kl, fm, disc/gen LSGAN, mel L1).
- [x] Phase 3d — `tiny_tts/audio.py` (load, STFT, mel basis, mel module,
  segment slicer that accepts arbitrary leading dims).
- [x] Phase 3e — `tiny_tts/data/dataset.py` + `tiny_tts/data/preprocess.py`
  (Thorsten metadata reader, on-disk spec/phone caches, padded collator).
- [x] Phase 3f — `tiny_tts/train.py` + `tiny_tts/utils/train_config.py`
  (single-GPU, AMP, both optimizers, MAS noise annealing, LR schedule,
  checkpoint save).
- [x] Phase 3g — `scripts/finetune_de.py`
  (loads English checkpoint, remaps embedding via the existing utility,
  delegates to `Trainer.run`).
- [x] Phase 4 — Browser multi-language support
  - `scripts/gen_metadata.py` writes the model sidecar JSONs.
  - `models/tinytts-en.json` (uses the saved 219-symbol snapshot — the shipped
    English ONNX was trained against that order).
  - `models/tinytts-de.json` (uses the new 220-symbol union).
  - `npm-package/g2p_en.js` extracted from the old `index.js`.
  - `npm-package/g2p_de.js` reads `g2p_de_rules.json`.
  - `npm-package/index.js` rewritten to be metadata-driven:
    `new TinyTTS({ modelPath, metadataPath })` picks G2P from `metadata.language`.
    Default English path stays backward-compatible (auto-falls-back to
    `models/tinytts-en.json` when no sidecar is found).
  - `npm-package/index.d.ts` — added `metadataPath` + `metadata` typings.
  - `npm-package/package.json` — added `g2p_en.js`, `g2p_de.js`,
    `g2p_de_rules.json` to the `files` array.
  - `npm-package/bin/cli.js` — new `--metadata` flag.
- [x] Phase 5 — `export_onnx.py` rewritten:
  - argparse, no Windows-hardcoded paths.
  - Writes the ONNX file *and* its `.json` sidecar.
  - For English exports it uses the snapshot symbol list automatically.
- [x] Bonus quality-of-life:
  - `tiny_tts/text/__init__.py` exposes `get_g2p(language)`.
  - `tiny_tts/infer.py --lang DE` switches G2P at the CLI (so the Phase 5
    inference verification step can exercise the German pipeline once a
    German checkpoint exists).

### Verified
- [x] G2P parity (805 words): pass.
- [x] New symbol table size: 220 (was 219). The single new symbol is `yː`.
- [x] JS modules load without syntax errors (`g2p_en.js`, `g2p_de.js`).
- [x] `index.js` end-to-end smoke test (with stubbed `onnxruntime-node` /
  `wavefile`): both EN and DE metadata paths produce the right `phoneIds`,
  `langIds`, and `toneIds` (DE: lang_id=7, tone_offset=14;
  EN: lang_id=2, tone_offset=7).
- [x] All new Python files parse cleanly (AST check).

### Not run in this session
- [ ] Smoke training run (`scripts/finetune_de.py --max-steps 100`):
  requires `torch` + audio deps + a Thorsten subset on disk. Code is
  written but not exercised here. Run on a GPU box.
- [ ] Inference parity (Python): needs a trained German checkpoint.
- [ ] ONNX export of the German checkpoint: ditto.
- [ ] Browser inference (`node bin/cli.js`): needs a published German ONNX
  + sidecar. The infrastructure is in place.

## Notes / Decisions
- The eager `import torch` in `tiny_tts/__init__.py` makes `import tiny_tts.text.german`
  pull torch. The build-time scripts (`scripts/gen_de_rules.py`,
  `scripts/gen_metadata.py`, `scripts/test_g2p_parity.py`) install a
  no-op `tiny_tts` package shim in `sys.modules` so they can run without torch.
- I did NOT modify the global `N_SPEAKERS` / `SPK2ID` in
  `tiny_tts/utils/config.py` — that would break English inference. The
  German training reads `N_SPEAKERS_DE` / `SPK2ID_DE` from
  `tiny_tts/utils/train_config.py` instead.
- The German speaker (`THORSTEN`) maps to ID 0, same slot as the existing
  English `MALE` speaker. After fine-tuning, the speaker embedding row is
  reused — the embedding shape (`emb_g`) is `[1, gin_channels]` for both.
- The `_FRONT_VOWELS` constant in `german.py` is currently unused but kept
  alongside `_BACK_VOWELS` to mirror the original plan and to make adding
  more context-sensitive rules trivial.
- Loss weights (mel=45, kl=1, dur=1) are per the standard VITS recipe and
  the values noted in PLAN_DE.md.

## Final file inventory
**Modified:**
- `tiny_tts/models/synthesizer.py` (+ `forward()` method, +88 lines)
- `tiny_tts/text/__init__.py` (+ `get_g2p()` helper)
- `tiny_tts/infer.py` (+ `--lang` flag)
- `npm-package/index.js` (rewritten — metadata-driven)
- `npm-package/index.d.ts` (added types)
- `npm-package/package.json` (files array)
- `npm-package/bin/cli.js` (added `--metadata`)
- `export_onnx.py` (rewritten — portable, sidecar-aware)

**New:**
- `tiny_tts/audio.py`
- `tiny_tts/losses.py`
- `tiny_tts/train.py`
- `tiny_tts/data/__init__.py`
- `tiny_tts/data/dataset.py`
- `tiny_tts/data/preprocess.py`
- `tiny_tts/models/discriminator.py`
- `tiny_tts/utils/train_config.py`
- `scripts/finetune_de.py`
- `scripts/gen_de_rules.py`
- `scripts/gen_metadata.py`
- `scripts/test_g2p_parity.py`
- `scripts/_run_js_g2p.js`
- `scripts/de_test_words.txt`
- `npm-package/g2p_en.js`
- `npm-package/g2p_de.js`
- `npm-package/g2p_de_rules.json`
- `models/tinytts-en.json`
- `models/tinytts-de.json`
