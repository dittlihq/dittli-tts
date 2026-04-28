# German TTS Training & Browser Deployment Plan

## Context

The user has forked TinyTTS — a 1.6M-parameter VITS-derived TTS model that runs in the browser via ONNX. The published model is English-only and the **training code has been stripped** from the repo (`VoiceSynthesizer` only has `infer()`, no `forward()`; no discriminator, no loss module, no dataset, no training loop).

The user wants to:
1. Add German support and train a German checkpoint
2. Avoid GPL contamination → cannot use `espeak-ng` / `phonemizer`
3. Run inference in the browser (Node.js/JS) for German
4. Be able to swap between English and German later by replacing the model file

Decisions (confirmed):
- **Dataset**: Thorsten Voice (~23h, single male speaker, CC0)
- **Init**: Fine-tune from existing `checkpoints/G.pth`
- **Training**: Full HiFi-GAN adversarial (MultiPeriodDiscriminator + feature matching)
- **Language switching**: Metadata sidecar bundled with each ONNX model

---

## Phase 1 — Symbol Table & Phoneme Inventory

**File:** [tiny_tts/text/symbols.py](tiny_tts/text/symbols.py#L246)

The combined symbol list is built as `sorted(set(zh + ja + en + … + de + ru))`, so **adding any new symbol shifts every phoneme ID alphabetically after it** — breaks fine-tuning unless we remap the embedding.

**Strategy**: maximize reuse of symbols already present from other languages, add only what's truly missing, and write a one-shot embedding remapper.

### German phoneme set (final)
```
Consonants: p b t d k g    f v s z ʃ ç x h    m n ŋ    l ʁ j    pf ts tʃ
Vowels:     i iː y yː ɪ ʏ  e eː ø øː ɛ ɛː œ   a aː     o oː ɔ u uː ʊ
Reduced:    ə ɐ
Diphthongs: aɪ aʊ ɔʏ   (emitted as 2-symbol sequences)
Optional:   ʔ            (glottal stop — skip for v1)
```

All but `y` and the long-vowel-with-ː pairs are already in the union of existing language symbol sets.

### Modifications
1. Append to `de_symbols`: `["y"]` (long ü). The existing `"y"` from `zh_symbols` collides — but since the encoder receives `language_id=7` per token, the embedding can disambiguate. To avoid the collision entirely we instead add `"yː"` as a single token (no conflict).
2. Add length-marked variants we'll emit: nothing — emit `vowel + ː` as 2 tokens, since `ː` already exists.
3. Update `de_symbols = ["ʏ", "̩", "yː"]` (one new symbol).

### Embedding remapper
**New file:** `tiny_tts/utils/remap_checkpoint.py`

Given old `symbols` and new `symbols`, build a lookup `old_idx → new_idx` for every overlapping symbol. For `enc_p.emb.weight` (shape `[old_n_vocab, hidden]` → `[new_n_vocab, hidden]`), copy rows for matching symbols, randomly init new rows. Used once before fine-tuning.

---

## Phase 2 — G2P Implementations (Python + JS, parallel)

Both implementations must produce **identical output** for any input — this is critical so the JS browser inference matches the Python training.

### 2a — Python G2P
**New file:** [tiny_tts/text/german.py](tiny_tts/text/german.py)

Public API mirrors [tiny_tts/text/english.py](tiny_tts/text/english.py):
```python
def g2p(text: str) -> tuple[list[str], list[int], list[int]]:
    """Returns (phones, tones, word2ph)."""
```

Internals:
- `normalize_text(text)`: expand German numbers (`123` → `einhundertdreiundzwanzig`), abbreviations (`Dr.` → `Doktor`, `z.B.` → `zum Beispiel`, `usw.` → `und so weiter`), times, dates. Reuse pattern from [tiny_tts/text/english_utils/](tiny_tts/text/english_utils/).
- `apply_rules(word)`: greedy longest-match scanner. Rule table is a Python list of `(pattern, output_phonemes_or_callable)`. Context-sensitive rules (`ch`, `st`/`sp`, `r`, `v`) are callables that inspect surrounding chars.
- `EXCEPTION_DICT`: ~500 common loanwords (Café, Virus, Computer, Genre, …) hardcoded in the file.
- All phonemes go through `_mapPhoneme()` (already exists in english.py logic) to ensure they're in the symbol table.

Tones for German: always `0` (single tone language, `num_de_tones = 1`).

### 2b — JavaScript G2P (browser)
**New file:** [npm-package/g2p_de.js](npm-package/g2p_de.js)

**Must be a 1:1 port of the Python implementation.** Same rule order, same exception dict, same normalizer. Approach:
- Maintain rules as a JSON file [npm-package/g2p_de_rules.json](npm-package/g2p_de_rules.json) — generated from the Python file by a small build script (`scripts/gen_de_rules.py`) so they cannot drift.
- Exception dict goes into the same JSON.
- The JS file imports the JSON and runs the same scanner.

Exports:
```js
module.exports = { graphemeToPhonemeDE };
// Returns { phones: [...], tones: [...], word2ph: [...] }
```

### 2c — Test parity harness
**New file:** `scripts/test_g2p_parity.py`

Runs Python G2P and shells out to Node to run JS G2P over the same word list (~1,000 words), asserts outputs are identical. Catches drift between the two implementations on every change.

---

## Phase 3 — Training Infrastructure

### 3a — `VoiceSynthesizer.forward()` (training pass)
**File:** [tiny_tts/models/synthesizer.py](tiny_tts/models/synthesizer.py#L666)

Insert before `infer()`. Mirrors Bert-VITS2 structure but uses this repo's component names:

```python
def forward(self, x, x_lengths, y, y_lengths, sid, tone, language, bert, ja_bert):
    # 1. encode prior from text
    x_enc, m_p, logs_p, x_mask, g = self.enc_p(x, x_lengths, tone, language, bert, ja_bert, sid=sid)
    # 2. encode posterior from spectrogram (training only)
    z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
    # 3. flow forward (z → z_p) for KL
    z_p = self.flow(z, y_mask, g=g)
    # 4. monotonic alignment search (negative-cross-entropy on z_p vs prior)
    with torch.no_grad():
        attn = mas_align(z_p, m_p, logs_p, x_mask, y_mask, noise_scale=self.current_mas_noise_scale)
    # 5. duration loss inputs (sum of attn columns = phoneme durations)
    w = attn.sum(2)
    l_length_sdp = self.sdp(x_enc, x_mask, w=w, g=g)            # NLL via flow
    l_length_dp = F.l1_loss(self.dp(x_enc, x_mask, g=g), torch.log(w + 1e-6) * x_mask)
    # 6. expand prior, slice random segment, decode to wav
    m_p_exp = torch.matmul(attn.squeeze(1), m_p.transpose(1,2)).transpose(1,2)
    logs_p_exp = torch.matmul(attn.squeeze(1), logs_p.transpose(1,2)).transpose(1,2)
    z_slice, ids_slice = commons.random_segments(z, y_lengths, self.segment_size)
    o = self.dec(z_slice, g=g)
    return o, l_length_sdp, l_length_dp, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)
```

Reuses existing components (`enc_p`, `enc_q`, `flow`, `sdp`, `dp`, `dec`) and existing utilities (`commons.random_segments`, `commons.kl_divergence`, `alignment.core.viterbi_decode_kernel`).

### 3b — Discriminator
**New file:** `tiny_tts/models/discriminator.py`

Standard HiFi-GAN `MultiPeriodDiscriminator` + `MultiScaleDiscriminator`. Periods `[2,3,5,7,11]`. ~3M parameters total (only used during training, dropped at inference). Direct adaptation from public HiFi-GAN code.

### 3c — Losses
**New file:** `tiny_tts/losses.py`

```python
def kl_loss(z_p, logs_q, m_p, logs_p, z_mask)         # uses commons.kl_divergence
def feature_matching_loss(fmap_r, fmap_g)
def discriminator_loss(disc_real, disc_gen)
def generator_loss(disc_gen)
def mel_loss(o, y_mel, mel_fn)                         # L1 on mel
```

### 3d — Audio / mel spectrogram
**New file:** `tiny_tts/audio.py`

```python
class MelSpectrogram(nn.Module):  # torchaudio.transforms.Spectrogram + MelScale
def load_audio(path, sr=44100) -> Tensor
def spectrogram_torch(y, n_fft=2048, hop=512, win_size=2048) -> Tensor
def spec_to_mel_torch(spec, n_fft, n_mels=128, sr=44100, fmin=0, fmax=None) -> Tensor
```

Standard torchaudio-based implementation. Hyperparameters from existing [tiny_tts/utils/config.py](tiny_tts/utils/config.py).

### 3e — Dataset
**New file:** `tiny_tts/data/dataset.py`

Thorsten Voice ships as `metadata.csv` (`filename|transcript`) + `wavs/*.wav` at 22050 Hz. Pipeline:
1. Resample to 44100 (model expects 44.1k, see `SAMPLING_RATE` in config) at preprocessing time, cache `.spec.pt` next to each `.wav`.
2. `__getitem__` returns `(phone_ids, tone_ids, lang_ids, spec, wav, sid=0)`.
3. Phone IDs come from the new `german.py` G2P → `phonemes_to_ids("DE", ...)`.
4. BERT/ja_bert tensors filled with zeros (matching how inference handles them in `infer.py`).
5. Collator pads variable-length sequences and reports lengths.

**New file:** `tiny_tts/data/preprocess.py`
Pre-computes specs and phoneme IDs for all utterances → speeds up training, fails loudly on bad data.

### 3f — Training loop
**New file:** `tiny_tts/train.py`

Single-GPU, AMP-enabled. Standard VITS training step:
1. Forward pass through `VoiceSynthesizer.forward()`.
2. Compute `mel_o = mel_fn(o)`, `mel_y = mel_fn(y_slice)`.
3. Discriminator step: `loss_d = discriminator_loss(D(y_slice), D(o.detach()))`.
4. Generator step: `loss_g = mel_loss + kl_loss + duration_losses + adv_loss + feat_match_loss`.
5. Backward, clip grads (use existing `commons.clip_grad_value_`), step both optimizers.
6. Anneal `current_mas_noise_scale` per `noise_scale_delta` (already a field on the model).
7. Save `G_*.pth` and `D_*.pth` every N steps.

**New file:** `tiny_tts/utils/train_config.py`
Hyperparameters not currently in config: `LEARNING_RATE = 2e-4`, `BETAS = (0.8, 0.99)`, `LR_DECAY = 0.999875`, `BATCH_SIZE = 16`, `TOTAL_STEPS = 100_000`, loss weights `c_mel=45`, `c_kl=1.0`, `c_dur=1.0`.

### 3g — Fine-tuning entrypoint
**New file:** `scripts/finetune_de.py`
1. Load `checkpoints/G.pth`.
2. Build new `VoiceSynthesizer` with the *new* (German-extended) `n_vocab`.
3. Apply `remap_checkpoint.py` to align embedding rows.
4. Initialize discriminator from scratch.
5. Run training loop.

---

## Phase 4 — Browser Multi-Language Support

### 4a — Model metadata sidecar
Each shipped ONNX bundle gets a tiny JSON:

**File:** `models/tinytts-de.json` (paired with `tinytts-de.onnx`)
```json
{
  "language": "de",
  "language_id": 7,
  "tone_offset": 21,
  "sample_rate": 44100,
  "symbols": [...],
  "phoneme_set": "german_v1"
}
```

`tone_offset` = `language_tone_start_map["DE"]` from [symbols.py:285](tiny_tts/text/symbols.py#L285). `symbols` is the exact symbol list the model was trained with — JS uses it to build its own `SYM` lookup, eliminating the hardcoded array currently at [npm-package/index.js:22](npm-package/index.js#L22).

### 4b — Refactor `TinyTTS` JS class
**File:** [npm-package/index.js](npm-package/index.js)

Changes:
1. Constructor: `new TinyTTS({ modelPath, metadataPath })` — `metadataPath` defaults to `modelPath.replace('.onnx', '.json')`.
2. `init()` reads metadata, picks G2P based on `language` field:
   ```js
   const G2P = { en: enGraphemeToPhoneme, de: deGraphemeToPhoneme }[meta.language];
   ```
3. `SYMBOLS`, `LANG_ID`, `TONE_OFFSET` come from metadata, not hardcoded constants.
4. Existing English G2P moves to [npm-package/g2p_en.js](npm-package/g2p_en.js) (extract from current `index.js`).
5. New [npm-package/g2p_de.js](npm-package/g2p_de.js) from Phase 2b.

Result: `tts = new TinyTTS({ modelPath: './tinytts-de.onnx' })` automatically picks German G2P. Replacing the `.onnx` (and its `.json`) is the only thing the user has to do.

### 4c — Package additions
[npm-package/package.json](npm-package/package.json) `files` array gains: `g2p_en.js`, `g2p_de.js`, `g2p_de_rules.json`. The existing `cmudict.json` and `g2p_predict.js` (English neural fallback) stay — German doesn't need them.

### 4d — CLI
[npm-package/bin/cli.js](npm-package/bin/cli.js): no functional changes, but the `--model` flag now drives language selection automatically via the sidecar.

---

## Phase 5 — ONNX Export

**File:** [export_onnx.py](export_onnx.py) — already exists for English.

Adapt to:
1. Load the German checkpoint.
2. Export the same 4 ONNX files (`text_encoder`, `duration_predictor`, `flow`, `decoder`) — or a single bundled `tinytts-de.onnx` to match what `index.js` currently expects (verify by re-reading the export script).
3. Write the metadata sidecar JSON next to the ONNX.

---

## Critical Files Modified vs Created

**Modified:**
- [tiny_tts/text/symbols.py](tiny_tts/text/symbols.py) — extend `de_symbols`
- [tiny_tts/text/__init__.py](tiny_tts/text/__init__.py) — register German G2P
- [tiny_tts/models/synthesizer.py](tiny_tts/models/synthesizer.py) — add `forward()` to `VoiceSynthesizer`
- [tiny_tts/utils/config.py](tiny_tts/utils/config.py) — `N_SPEAKERS`, `SPK2ID` for Thorsten
- [npm-package/index.js](npm-package/index.js) — metadata-driven G2P dispatch
- [npm-package/package.json](npm-package/package.json) — file list

**Created:**
- `tiny_tts/text/german.py`, `tiny_tts/text/german_utils/` (number/abbrev/time normalization)
- `tiny_tts/utils/remap_checkpoint.py`
- `tiny_tts/models/discriminator.py`
- `tiny_tts/losses.py`, `tiny_tts/audio.py`
- `tiny_tts/data/dataset.py`, `tiny_tts/data/preprocess.py`
- `tiny_tts/train.py`, `tiny_tts/utils/train_config.py`
- `scripts/finetune_de.py`, `scripts/gen_de_rules.py`, `scripts/test_g2p_parity.py`
- `npm-package/g2p_en.js` (extracted), `npm-package/g2p_de.js`, `npm-package/g2p_de_rules.json`
- `models/tinytts-en.json`, `models/tinytts-de.json` (sidecars)

---

## Verification

1. **G2P parity**: `python scripts/test_g2p_parity.py` — Python and JS must produce identical phoneme sequences on a 1000-word German test list.
2. **Symbol table integrity**: `python -c "from tiny_tts.text.symbols import symbols; print(len(symbols))"` — record the new size; ensure `enc_p.emb` matches it after remapping.
3. **Training smoke test**: run `scripts/finetune_de.py --max-steps 100` on a 10-utterance subset → loss curves must decrease, no NaN, checkpoints save.
4. **Full fine-tune**: ~50–100k steps on Thorsten (~24h on a single A100, ~3 days on a 3090).
5. **Inference parity (Python)**: `python -m tiny_tts.infer --lang DE --text "Guten Morgen, wie geht es dir?"` produces intelligible German.
6. **ONNX export**: `python export_onnx.py --checkpoint G_de.pth --out models/tinytts-de.onnx` — check file size matches expected ~6MB FP16.
7. **Browser inference**: from the `npm-package/` dir, `node bin/cli.js "Guten Morgen" --model ../models/tinytts-de.onnx -o out.wav` → audible German speech.
8. **Language switching**: same Node script, `--model ../models/tinytts-en.onnx "Hello"` → audible English. No code changes between the two runs.
9. **Type checking**: `npm run build` (or `tsc --noEmit`) passes against the updated `index.d.ts`.

---

## Out of Scope / Risks

- **Sample rate mismatch**: Thorsten is 22050 Hz, model is 44100 Hz. Resampling at preprocessing is straightforward but loses no info; aliasing would only matter going the other direction.
- **G2P quality on rare loanwords**: ~5% of words may have minor phoneme errors. Acceptable — the model learns to smooth over consistent G2P quirks.
- **No glottal stop / vocalic-r polish in v1**: vocalic `r` (`ɐ` for syllable-final `r`) is included; glottal stop `ʔ` is not. Adds ~1% naturalness — defer to v2.
- **Single-speaker only**: Thorsten is single-speaker, so `N_SPEAKERS = 1` and `SPK2ID = {"THORSTEN": 0}`. Multi-speaker requires more changes (style encoder pretraining).
- **No BERT features**: training will pass zeros for `bert`/`ja_bert` (matching inference). The model already supports this — the BERT projection layers will simply produce constant output and be effectively no-ops.
