# Adding a language

This is the end-to-end recipe for adding a new language to Dittli TTS. The goal
is that you can follow it without reading the engine internals.

Each language is an independent **pack** (`@dittli/tts-<code>`) shipping
`{metadata, model.onnx}` plus whatever its G2P needs. Python (`src/dittli_tts/`)
owns the symbol table and trains/exports the model; the npm pack consumes the
generated artifacts.

> **Quick start:** `python scripts/new_language.py <code> --name "<Name>"`
> scaffolds the pack skeleton and prints this checklist with your code filled in.

---

## 0. Pick your G2P strategy

The grapheme→phoneme step is the only genuinely language-specific work. Choose
based on the language's orthography:

| Strategy | Use when | What you write | What ships |
|----------|----------|----------------|------------|
| **Rule-based** (like German) | spelling→sound is mostly regular | a Python rule G2P + a 1:1 JS port + a parity test | a small rules JSON |
| **Neural ONNX** (like English) | orthography is irregular / large | train a small seq2seq G2P, export with `scripts/export_g2p_onnx.py` | `g2p_encoder.onnx` + `g2p_decoder_step.onnx` + `g2p_vocab.json` |
| **Hybrid** (English today) | common words regular, long tail irregular | a lookup dict for common words + a neural ONNX fallback | dict + the ONNX graphs |

The neural-ONNX path reuses the generic host loop (`createOnnxG2p` in
`@dittli/tts-core`), so **you do not hand-port a G2P to JS and you do not write a
parity suite** — you ship weights + a vocab sidecar and the runtime drives them.
That is the recommended path for any language without clean pronunciation rules.

> After Step 2 of `docs/2026-06-01_PLAN_G2P_INTEGRATION.md` lands (the unified
> char front-end), the G2P column disappears for new languages entirely — the
> model takes characters and a pack is just `{metadata, model.onnx}`.

---

## 1. Register the language (symbols, tone, id)

In `src/dittli_tts/text/symbols.py`:
- add any phoneme symbols your G2P emits that aren't already in the table,
- add the language's tone count / tone offset and a language id.

Then regenerate the npm-side sidecars (Python is the source of truth):

```bash
npm run g2p:metadata          # python scripts/gen_metadata.py
```

## 2. Provide the G2P

**Rule-based:** add `src/dittli_tts/text/<lang>.py` (mirror `german.py`:
`normalize_text` + `grapheme_to_phoneme`), the JS port
`packages/tts-<code>/src/g2p_<code>.js`, and regenerate the rules JSON:

```bash
npm run g2p:gen               # python scripts/gen_de_rules.py-style generator
```

Add a parity test under `tests/parity/` (see `test_g2p_de_parity.py`) so the
Python and JS implementations can't drift.

**Neural ONNX:** train a seq2seq G2P for the language, then bake it:

```bash
uv run --extra dev --with onnx --with onnxruntime \
    python scripts/export_g2p_onnx.py --assets --fp16 --verify
```

This writes `g2p_encoder.onnx`, `g2p_decoder_step.onnx`, and `g2p_vocab.json`
into the pack. Wire the pack's G2P to `createOnnxG2p` (copy the English pack's
`prepare()` + OOV path from `packages/tts-en/src/g2p_en.js`). No JS port, no
parity suite — verify with `scripts/_g2p_onnx_parity.mjs`.

## 3. Data + fine-tune

Warm-start from an existing checkpoint (don't train from scratch):

```bash
bash scripts/setup_<lang>_data.sh        # fetch a single-speaker dataset
python -m dittli_tts.data.preprocess \
    --metadata data/<lang>/metadata.csv --wavs-dir data/<lang>/wavs --language <LANG>
python scripts/finetune_<lang>.py \
    --metadata ... --wavs-dir ... --english-ckpt checkpoints/G.pth --ckpt-dir checkpoints_<lang>/
```

See `docs/2026-04-28_TRAINING_DE.md` for the cloud-training guide (Modal/Kaggle).

## 4. Export the model

```bash
python -m dittli_tts.inference.export --lang <LANG> --checkpoint checkpoints_<lang>/G.pth \
    --out packages/tts-<code>/assets/<code>/model.onnx
```

(FP16 is produced automatically; the pack ships the FP16 graph.)

## 5. Package + smoke test

The scaffold already created `packages/tts-<code>/` with `package.json`,
`src/index.js` (registers the pack), and `src/index.d.ts`. Confirm:

```bash
node scripts/check-publish-assets.js tts-<code>   # all required files present
npm run test:js                                    # JS suite green
```

Add a smoke test that synthesises a sentence in the new language (mirror the
existing pack tests), and you're done.

---

## Checklist

- [ ] symbols + tone offset + language id in `symbols.py`; `npm run g2p:metadata`
- [ ] G2P: rule-based (+ JS port + parity test) **or** neural ONNX (`export_g2p_onnx.py`)
- [ ] dataset fetched; fine-tune warm-started from an existing checkpoint
- [ ] `model.onnx` exported into the pack
- [ ] pack registers via `registerLanguagePack`; `check-publish-assets` passes
- [ ] smoke test synthesises the new language
