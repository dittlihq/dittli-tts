# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the npm packages
(`@dittli/tts-core`, `@dittli/tts-en`, `@dittli/tts-de`) follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For the narrative behind these releases — decisions, dead ends, and gotchas —
see [`docs/HISTORY.md`](docs/HISTORY.md).

## [Unreleased]

## [0.6.0] — 2026-06-09

### Fixed
- **`@dittli/tts-en` / `@dittli/tts-de`: acronyms and runs of consecutive
  uppercase letters are now spelled out** (e.g. "NASA", "API", "USB") instead
  of producing near-silent or garbled speech. Initialisms are expanded to their
  spoken letters before phonemisation, in both the JS G2P and the Python
  front-end.
- **`@dittli/tts-core`: stop fetching the 25 MB JSEP WASM at runtime.**
  `src/ort.js` imported the default `onnxruntime-web` entry, whose loader
  hard-references `ort-wasm-simd-threaded.jsep.wasm` (~25 MB, WebGPU build)
  even though we only use the CPU `wasm` provider and only ship the non-jsep
  binary. Switched to the `onnxruntime-web/wasm` subpath, which fetches the
  non-jsep `ort-wasm-simd-threaded.wasm` (~12.4 MB) — the binary
  `copy-ort-wasm.js` already ships — and pulls a smaller JS bundle
  (0.39 MB → 0.07 MB). FP16 model is unaffected (stays 4.6 MB).
- **`@dittli/tts-core`: silence ORT's `[W:onnxruntime:...]` graph-optimisation
  warnings.** They route through `console.error`, so the old `console.warn`
  filter never caught them; raised `ort.env.logLevel` (and the per-session
  `logSeverityLevel`) to `error` instead. Pass `{ verbose: true }` to keep them.

### Changed
- `@dittli/tts-core`: bumped `onnxruntime-web` dependency to `^1.26.0`.
- Docs: renamed `docs/` files to carry their creation date as a prefix,
  renamed `SESSION_NOTES.md` → `docs/HISTORY.md` (now a full project history),
  and added this changelog.

## [0.5.0] — 2026-05-19

### Added
- New public API on `@dittli/tts-core`: `new DittliTTS({ language, assetBase })`,
  `play()`, `synthesize()`, `stop()`, `dispose()`, `loadLanguage()`, and the
  `DittliTTS.preloadWhenIdle()` static factory. Exposes `AudioContextLockedError`
  and `floatToWav()` for power users.
- `.d.ts` type declarations shipped from every package (no more `declare module`
  shim).
- Vitest + happy-dom test suite (`tests/js/`), 58/58 passing.

### Changed
- **Clean-break browser API reshape.** Consumer setup collapses to ~10 lines
  plus a one-time copy of each pack's `assets/` + `ort-wasm/` into the static
  tree.
- Assets relocated to `assets/<lang>/` (renames preserve git history).
- `@dittli/tts-core` is now a `peerDependency` (`^0.5.0`) of the language packs.
- `src/ort.js` is the single `onnxruntime-web` touchpoint.

### Removed
- `static register*`, `modelUrl`/`metadataUrl` constructor options,
  `speak()` returning WAV bytes, and the public `textToPhonemeIds` — no shim.
- Intent to drop the JSEP/WebGPU WASM variant from the publish copy step.
  (Note: the default-entry import still pulled the jsep binary at runtime until
  the Unreleased fix above.)

## [0.4.0] — 2026-05-18

### Changed
- Clean asset layout and consumer ergonomics overhaul.
- Removed module-level `new URL(..., import.meta.url)`, which had blocked
  English under Vite dependency pre-bundling.

_Not published to npm (skipped in favour of the 0.5.0 clean break)._

## [0.2.0] — 2026-04-30

### Added
- Last public npm release prior to the 0.5.0 clean break.

## [0.1.0] — 2026-04-30

### Added
- Split the single package into an npm workspace: `@dittli/tts-core`
  (ONNX inference engine + CLI), `@dittli/tts-en` (English G2P, CMU dict,
  neural fallback), and `@dittli/tts-de` (German rules).
- German language support (Thorsten Voice): symbol-table extension, Python + JS
  G2P (parity-tested), training infrastructure, and ONNX export.
- Repository restructure (TinyTTS → Dittli TTS): `src/` layout, `pyproject.toml`
  + `uv`, Biome + Ruff linting, and a pytest suite.

[Unreleased]: https://github.com/dittlihq/dittli-tts/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/dittlihq/dittli-tts/releases/tag/v0.5.0
[0.4.0]: https://github.com/dittlihq/dittli-tts/compare/v0.2.0...v0.4.0
[0.2.0]: https://github.com/dittlihq/dittli-tts/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/dittlihq/dittli-tts/releases/tag/v0.1.0
