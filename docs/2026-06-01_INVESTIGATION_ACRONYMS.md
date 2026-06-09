# Investigation: acronyms / consecutive uppercase letters produce near-silent or garbled speech

## Symptom

Input containing runs of uppercase letters — e.g. **"TV-Serie"** (DE) or
**"FBI"**, **"GPU"** (EN) — synthesises to almost nothing, or to garbled
audio. Reported for both languages; the same class of bug exists in the
upstream TinyTTS project.

## Root cause

The G2P front-end **lowercases the text before phonemisation and has no
initialism handling**, so an initialism is fed to the word-level G2P as if it
were an ordinary word. The result is a phoneme string that is too short
(often vowel-less) to synthesise audibly.

Ground-truth phoneme output on `develop` (padding stripped):

| Input | Lang | Phonemes (before fix) | Problem |
|-------|------|-----------------------|---------|
| `TV` | DE | `t f` | two consonants, **no vowel** → near-silent |
| `TV-Serie` | DE | `t f s e ʁ i ː` | "TV" part inaudible |
| `USA` | DE | `u ɛ s a`* | letters dropped/merged |
| `FBI` | EN | `b ay` | sounds like "bye" |
| `GPU` | EN | `jh uw` | sounds like "joo" |
| `PDF` | EN | `d eh f` | "def" |
| `HTTP` | EN | `t ae p t iy` | nonsense |

(\* DE "USA" before fix lost the leading letter entirely.)

### Mechanism, with file references

1. **English** — `src/dittli_tts/text/english.py` `normalize_text()` calls
   `text.lower()` as its first step. `grapheme_to_phoneme()` then looks each
   token up in the CMU dict (`w.upper()`) or falls back to the neural g2p_en
   predictor. An initialism like `FBI` becomes the lowercase token `fbi`,
   which the neural model "pronounces" as a single short word.
2. **German** — `src/dittli_tts/text/german.py`: the rule scanner
   (`_apply_rules`) lowercases each word, then applies grapheme rules.
   `TV` → `tv` → `t` + (`v_rule` →) `f` = `['t', 'f']`. German letter `v` maps
   to `/f/` and there is no vowel, so the synthesiser has nothing to lengthen.
3. There was **no step anywhere that detects consecutive uppercase letters**
   and spells them out. Capitalised German nouns ("Serie") are single
   capitals and are not the problem — only runs of **two or more** are.

## Fix

Add an **initialism expansion** step that runs *before* lowercasing: each run
of two or more uppercase letters is replaced with the space-separated spoken
**letter-name** spelling for the language, which the existing G2P then
pronounces correctly as words.

- `TV` → `te vau` (DE) / `tee vee` (EN)
- `FBI` → `ef be i` (DE) / `ef bee eye` (EN)
- `TV-Serie` → `te vau -Serie` → `t e f a ʊ` + `s e ʁ i ː` (the word survives)

New single-source-of-truth module:
[`src/dittli_tts/text/initialisms.py`](../src/dittli_tts/text/initialisms.py)
holds the per-language letter tables and `expand_initialisms(text, lang)`.
Wired into both `normalize_text` functions. The JS packages
(`packages/tts-en/src/g2p_en.js`, `packages/tts-de/src/g2p_de.js`) carry
hand-ported copies, kept honest by the German parity test (acronyms were
added to `scripts/de_test_words.txt`).

### Output after fix

| Input | Lang | Phonemes (after fix) |
|-------|------|----------------------|
| `TV` | DE | `t e f a ʊ` (te-vau) |
| `TV-Serie` | DE | `t e f a ʊ s e ʁ i ː` (te-vau + serie) |
| `FBI` | EN | `eh f b iy ay` (ef-bee-eye) |
| `GPU` | EN | `jh iy p iy y uw` (jee-pee-you) |
| `USA` | EN | `y uw eh s ey` (you-es-ay) |

## Trade-off (intentional, documented)

Acronyms conventionally read as a single word — **NASA**, **NATO** — are now
also spelled out ("en-ay-es-ay"). Spelling out is the safer default: it is
intelligible, whereas the previous behaviour was silent/garbled, and the vast
majority of all-uppercase tokens users type are true initialisms. A future
refinement could add a small allowlist of word-acronyms that bypass expansion.

## Validation

- New unit tests: [`tests/unit/test_initialisms.py`](../tests/unit/test_initialisms.py)
  (expansion behaviour + every letter name yields a vowel + the "TV-Serie"
  regression).
- German Python↔JS parity confirmed on `TV`, `TV-Serie`, `FBI`, `GPU`, `USA`,
  `NASA`, `ABC`, `USB` (identical output).
- Full suites: `uv run pytest` and `npm run test:js` green; ruff + biome clean.
