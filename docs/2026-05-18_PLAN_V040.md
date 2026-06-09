# dittli-tts v0.4.0 — API reshape plan

Clean-break successor to v0.2.0. Replaces the consumer's 75-LOC integration
workaround with a ~10-line setup. No 0.3.x deprecation shims — everything
back-compat-breaking lands in this one release.

## Why this exists

After integrating `@dittli/tts-{core,en,de}@0.2.0` into the Dittli Angular
app, the consumer documented (1) German works after a non-trivial workaround
(~75 LOC + six asset-config entries + a TS declaration shim), and (2)
**English is completely blocked** because g2p assets (`cmudict.json`,
`g2p_model.json`) are loaded via `new URL("./cmudict.json", import.meta.url)`
at module load time — which breaks under Vite's dep pre-bundling because
sibling asset folders aren't copied into `node_modules/.vite/deps/`.

The same `import.meta.url` failure cascades through `onnxruntime-web`'s
own WASM resolution, forcing the consumer to install `onnxruntime-web`
directly and configure `ort.env.wasm.wasmPaths` to a path they control.

This reshape:
1. Unblocks English by killing all `new URL(..., import.meta.url)` at
   module-load time. Assets are resolved at fetch time from a single
   consumer-provided `assetBase`.
2. Makes core own all ONNX Runtime setup (WASM paths, warning filter,
   session lifecycle, AbortSignal plumbing).
3. Collapses the WAV-bytes → Blob → object-URL → `<audio>` dance into
   `await tts.play(text)`.
4. Ships TypeScript declarations from language packs so TS strict-mode
   consumers don't need a manual `declare module` shim.

## Target consumer code (the success metric)

```ts
@Injectable({ providedIn: "root" })
export class TtsService {
    private db = inject(DatabaseService);
    private tts = new DittliTTS({
        language: this.db.ttsLanguage(),
        assetBase: "/tts/",
    });
    private controller: AbortController | null = null;

    public preload(): void {
        DittliTTS.preloadWhenIdle({
            language: this.db.ttsLanguage(),
            assetBase: "/tts/",
        });
    }

    public async speak(text: string): Promise<void> {
        this.controller?.abort();
        this.controller = new AbortController();
        await this.tts.play(text, { signal: this.controller.signal });
    }
}
```

Consumer build config collapses to:
- Copy `node_modules/@dittli/tts-{de,en}/assets/` → `public/tts/`
- Copy `node_modules/@dittli/tts-core/ort-wasm/` → `public/tts/ort/`

No `import * as ort from "onnxruntime-web"` anywhere in consumer code.
No manual `LANG_PACKS` URL bookkeeping. No Blob/object-URL plumbing.
No engine cache. No warning filter. No TS declaration shim.

## New public API

### Types

```ts
export type LanguageCode = string; // BCP-47 or 2-letter; normalized internally

export interface AssetLayout {
  /** Required: base URL prefix for per-language assets. Trailing "/" is normalized. */
  assetBase: string;
  /** Optional: separate base for ORT WASMs. Defaults to `${assetBase}ort/`. */
  ortAssetBase?: string;
}

export interface ProgressEvent {
  asset: "model" | "metadata" | "cmudict" | "g2p_model";
  language?: string;
  loaded: number;
  total: number; // 0 if Content-Length unknown
}

export interface DittliTTSOptions extends AssetLayout {
  /** Primary language. Accepts "de", "de-DE", "en-US", etc. Required. */
  language: LanguageCode;
  /** Default false. When false, library installs a console.warn filter for [W:onnxruntime] noise. */
  verbose?: boolean;
  /** Defaults to ["wasm"]. */
  executionProviders?: string[];
  /** Per-asset download progress. */
  onProgress?: (e: ProgressEvent) => void;
  /** Default false. When true, init() skips the kernel-warmup inference. */
  skipWarmup?: boolean;
}

export interface SpeakOptions {
  /** Override the instance's primary language for this call (multi-lang apps). */
  language?: LanguageCode;
  /** Cancels in-flight asset fetches AND aborts the ORT session.run (see "AbortSignal semantics"). */
  signal?: AbortSignal;
  speed?: number;       // default 1.0
  speaker?: string;     // metadata.spk2id lookup
}

export interface DecodedAudio {
  samples: Float32Array;
  sampleRate: number;
}

export class DittliTTS {
  constructor(opts: DittliTTSOptions);

  /** Idempotent. Loads model+metadata+g2p for the primary language and runs a kernel warmup (unless skipWarmup). */
  init(): Promise<void>;

  /** Add a second (or third) language to an already-constructed instance. */
  loadLanguage(language: LanguageCode, opts?: { signal?: AbortSignal }): Promise<void>;

  /** Pure inference. Returns decoded float samples + sample rate. */
  synthesize(text: string, opts?: SpeakOptions): Promise<DecodedAudio>;

  /** Owns an internal AudioContext; resolves when playback ends (or signal aborts). */
  play(text: string, opts?: SpeakOptions): Promise<void>;

  /** Stop any current playback without disposing the engine. */
  stop(): void;

  /** Release all ORT sessions and close the AudioContext. */
  dispose(): Promise<void>;

  /**
   * Construct + init + warm in one call, scheduled via requestIdleCallback
   * (falls back to setTimeout(0) on Safari). Returns the live instance.
   */
  static preloadWhenIdle(opts: DittliTTSOptions): Promise<DittliTTS>;
}
```

### `assetBase` resolution table

`assetBase` is a URL prefix (we normalize the trailing `/`). Per-asset paths:

| Asset                       | Resolved URL                          | Source today                                |
| --------------------------- | ------------------------------------- | ------------------------------------------- |
| Metadata                    | `${assetBase}${lang}/metadata.json`   | `packages/tts-XX/metadata/dittli-XX.json`   |
| ONNX model                  | `${assetBase}${lang}/model.onnx`      | `packages/tts-XX/model/dittli-XX_fp16.onnx` |
| CMU dict (en)               | `${assetBase}en/cmudict.json`         | `packages/tts-en/src/cmudict.json`          |
| G2P weights (en)            | `${assetBase}en/g2p_model.json`       | `packages/tts-en/src/g2p_model.json`        |
| ORT WASM (default)          | `${assetBase}ort/`                    | n/a (ORT internal)                          |
| ORT WASM (with override)    | `${ortAssetBase}`                     | n/a                                         |

Renaming the publishable artifacts to flat `metadata.json` / `model.onnx`
(instead of `dittli-de_fp16.onnx`) is the load-bearing simplification:
the layout becomes `cp -r node_modules/@dittli/tts-{core,de,en}/assets`
into one tree.

German g2p rules (`g2p_de_rules.json`) stay inlined via JSON import
attributes — they're tiny and bundle cleanly, no benefit to externalizing.

### Language-pack shape

```ts
// @dittli/tts-de/src/index.js
import { registerLanguagePack } from "@dittli/tts-core/internal";
import { graphemeToPhonemeDE } from "./g2p_de.js";

export const dePack = {
  language: "de",
  g2p: graphemeToPhonemeDE,
  // Relative paths; assetBase resolved at fetch time, not at module load.
  assets: { metadata: "de/metadata.json", model: "de/model.onnx" },
};
registerLanguagePack(dePack);  // side-effecting register on import
```

Two consumer paths, both supported:

```ts
// Path A: import-and-go (default, what we document)
import { DittliTTS } from "@dittli/tts-core";
import "@dittli/tts-de";  // side-effect register
const tts = new DittliTTS({ language: "de", assetBase: "/tts/" });

// Path B: explicit (for power users avoiding side-effect imports)
import { DittliTTS } from "@dittli/tts-core";
import { dePack } from "@dittli/tts-de";
const tts = new DittliTTS({ language: "de", assetBase: "/tts/", packs: [dePack] });
```

The registry lives inside `internal.js` (renamed `_LANG_REGISTRY`), NOT on
the public `DittliTTS` class. The integration report's "spooky action at a
distance" complaint is mitigated by (1) the registry being internal,
(2) the pack export being a no-magic alternative.

## Per-package change list

### `@dittli/tts-core`

**Add:**
- `src/ort.js` — wraps `import * as ort from "onnxruntime-web"`. Exports
  `configureOrt({ wasmPaths, verbose })`, `createSession(bytes, opts)`,
  `runSession(session, feeds, signal)`, `tensor(type, data, shape)`,
  `releaseSession(session)`. This is the **only** file in the repo that
  touches `ort.env`. The `console.warn` filter installs here, exactly
  once per page, via a module-level `_filterInstalled` flag.
- `src/audio.js` — `floatToWav(samples, sampleRate)` (relocated from
  `index.js`, kept exported for power users) and `playSamples({ samples,
  sampleRate, signal, audioContext })` that owns an `AudioBufferSourceNode`.
  The `AudioContext` is created lazily on first `play()` and stashed on
  the `DittliTTS` instance — not a module global.
- `src/internal.js` — `_LANG_REGISTRY`, `registerLanguagePack(pack)`,
  `_resolveAsset(base, rel)`, `_normalizeLanguage(tag)` (BCP-47 →
  2-letter: `de`/`de-DE`/`de-AT`/`de-CH` → `de`).
- `src/engine.js` — per-language `Engine { session, metadata, _symbolSet,
  pack }`. Owns model + metadata fetch, pack `prepare()`, session create,
  `textToPhonemeIds`, and the inference call (with AbortSignal via
  `Promise.race` against `session.run`).
- `src/index.d.ts` — full rewrite to the types above.
- `src/internal.d.ts` — types for `registerLanguagePack` + `LanguagePack`.

**Rewrite:**
- `src/index.js` — becomes thin orchestration: constructor, `init()`,
  `loadLanguage()`, `synthesize()`, `play()`, `stop()`, `dispose()`,
  `preloadWhenIdle()`. Delegates to the new modules.

**Plumbing decisions:**
- **Engine cache**: `Map<normalizedLang, Engine>` on the `DittliTTS`
  instance, populated by `loadLanguage()` (which `init()` calls once for
  the primary language). Idempotent via in-flight
  `Map<lang, Promise<Engine>>`.
- **AbortSignal**: `fetch(url, { signal })` handles asset cancellation
  natively. ORT doesn't support `signal` on `session.run` — we do
  `Promise.race([session.run(feeds), abortPromise])` and document that
  this releases the JS reference but doesn't actually cancel the
  WASM-side compute; the next `synthesize` call queues behind it.
  Aborting saves caller time, not CPU. **This is the riskiest unknown
  in the implementation — spike it first** (see "Implementation order").
- **AudioContext owner**: the `DittliTTS` instance. `play()` lazily
  constructs it. If the browser refuses construction (autoplay policy
  outside a user gesture), throw a typed error
  (`AudioContextLockedError`). `dispose()` closes it.
- **ORT warning filter**: installed by `configureOrt()` in `src/ort.js`
  behind `verbose: false` (the default). Globally monkey-patching
  `console.warn` from a library is normally a sin — be loud about it
  in the README; `verbose: true` opts out.
- **Kernel warmup**: end of `init()` runs `synthesize("a", { language:
  primary })` with the result discarded. Gated by `skipWarmup: true`
  for tests/CI.

**Vanishes (clean break, no shim):**
- `DittliTTS.registerLanguage` / `registerDefaultMetadata` /
  `registerDefaultModel` static methods.
- Constructor `modelUrl` / `metadataUrl` options.
- The auto-pick-the-one-registered-language fallback in `_init()`.
- `speak()` returning `Uint8Array` WAV bytes.
- `textToPhonemeIds` on the public surface (moves to `Engine`, exported
  from `./internal` for power users / parity tests).
- `_floatToWav` (private) becomes `floatToWav` (exported from `audio.js`).

### `@dittli/tts-de`

**Add:**
- `assets/de/metadata.json` (relocated from `metadata/dittli-de.json`)
- `assets/de/model.onnx` (relocated from `model/dittli-de_fp16.onnx`)
- `src/index.d.ts`
- `src/pack.js` (or keep `dePack` in `index.js`)

**Rewrite:**
- `src/index.js` — drop the three `new URL(..., import.meta.url)` calls.
  Body becomes the dePack export + `registerLanguagePack(dePack)` call.

**Delete:**
- `metadata/` and `model/` directories (assets moved to `assets/de/`).

**package.json changes:**
- `version: "0.4.0"`
- `files: ["src/**/*.js", "src/**/*.d.ts", "src/**/*.json", "assets/**"]`
- `sideEffects: ["./src/index.js"]` (the side-effecting register call)
- `exports`: add `"./pack": { "types": "./src/pack.d.ts", "import": "./src/pack.js" }`
  and `"./assets/*": "./assets/*"`
- Move `@dittli/tts-core` from `dependencies` to `peerDependencies` (`^0.4.0`)
  — prevents a hoisted layout from pulling a second copy of core.

### `@dittli/tts-en`

**Add:**
- `assets/en/metadata.json`, `assets/en/model.onnx`, `assets/en/cmudict.json`,
  `assets/en/g2p_model.json` (all relocated)
- `src/index.d.ts`
- `src/pack.js`

**Rewrite:**
- `src/g2p_en.js` — remove `CMU_URL = new URL(...)`. `prepare({ assetBase,
  signal, onProgress })` does the fetch. The neural inference functions
  are unchanged.
- `src/g2p_predict.js` — same: remove `MODEL_URL = new URL(...)`.
  `prepare({ assetBase, signal, onProgress })` does the fetch.
- `src/index.js` — export `enPack` + `registerLanguagePack(enPack)`.

**Delete:** `metadata/`, `model/`, `src/cmudict.json`, `src/g2p_model.json`
(relocated to `assets/en/`).

**package.json:** same shape as `tts-de`.

### How `prepare` sees `assetBase`

Core calls `pack.g2p.prepare({ assetBase, signal, onProgress })` from
inside `Engine.load()`, not at module-load time. That object is the only
contract; each pack decides which sub-URLs it needs.

## Packaging — ORT WASM strategy

**Re-export, don't bundle.** `@dittli/tts-core` re-exports the ORT WASMs
under `./ort-wasm/*`. The default `ortAssetBase` resolves to
`${assetBase}ort/`. The consumer copies
`node_modules/@dittli/tts-core/ort-wasm/` into their static asset tree
(one line of build config).

Why not bundle inline (base64):
- The WASMs are ~3-5 MB each; inlining defeats `Cache-Control` and bloats
  first paint.
- ORT picks its WASM at runtime based on SIMD/threads/JSEP feature
  detection — bundling all variants is wasteful.

Why not fetch from a CDN by default:
- Offline-capable apps (the Android WebView target) need local hosting.

**`tts-core/package.json` exports:**

```json
{
  "version": "0.4.0",
  "exports": {
    ".": { "types": "./src/index.d.ts", "import": "./src/index.js" },
    "./internal": { "types": "./src/internal.d.ts", "import": "./src/internal.js" },
    "./ort-wasm/*": "./ort-wasm/*"
  },
  "files": ["src/**/*.js", "src/**/*.d.ts", "ort-wasm/**"],
  "sideEffects": false
}
```

The `ort-wasm/` directory is populated by a `prepublishOnly` step
(`scripts/copy-ort-wasm.js`) that copies the needed subset from
`node_modules/onnxruntime-web/dist/`:
`ort-wasm-simd-threaded.{mjs,wasm}` and the `.jsep.*` variants.
This avoids re-vendoring ORT's whole `dist/`.

Add `ort-wasm/` to `.gitignore`.

## Implementation order

The riskiest unknown is **AbortSignal through ORT's `session.run`**.
Spike it first; everything else is mechanical.

1. **Spike (1 day): AbortSignal through ORT.** Prove out the
   `Promise.race` pattern against a real `InferenceSession.run` call in
   `examples/browser-vite/`. Specifically verify: (a) the JS promise
   rejects promptly; (b) the next `run()` call queues correctly without
   WASM-side corruption; (c) `release()` after an aborted run doesn't
   deadlock. If `Promise.race` isn't safe, fall back to "AbortSignal
   only cancels pre-`run()` fetches; document that `synthesize` is
   non-interruptible once tensors are in the WASM heap." Either way
   the API stays the same.
2. **`src/ort.js`** — extract all ORT touchpoints into one module. Add
   the `console.warn` filter behind `verbose`. No behavior change yet.
   Land on its own commit.
3. **`src/internal.js`** — `_LANG_REGISTRY`, `registerLanguagePack`,
   `_resolveAsset`, `_normalizeLanguage`.
4. **`src/audio.js`** — extract `floatToWav`, add `playSamples`.
5. **`src/engine.js`** — per-language `Engine`.
6. **`src/index.js`** — rewrite as thin orchestration. Add
   `synthesize()`, `play()`, `preloadWhenIdle()`. Drop `speak()`.
   Drop the static `register*` methods. Drop `modelUrl`/`metadataUrl`.
7. **`src/index.d.ts` + `src/internal.d.ts`** — full type rewrite.
8. **`tts-core/package.json` + ORT WASM copy script** — exports,
   `prepublishOnly`, `.gitignore`.
9. **`tts-de` relocation + rewrite + `.d.ts` + `package.json`.**
10. **`tts-en` relocation + rewrite + `.d.ts` + `package.json`** —
    g2p_en.js and g2p_predict.js lose their module-level URLs.
11. **`scripts/check-publish-assets.js`** — update manifest to point
    at new asset paths.
12. **`examples/browser-vite/`** — minimal Vite smoke app that
    exercises `init() → play()` end-to-end. Documents the consumer
    copy step. Keep it as the canonical smoke check.
13. **README update** — show before/after consumer code.

## Decisions / pushbacks

These deviate from the integration report; document the reasoning in
the changelog so the next maintainer doesn't relitigate.

- **Return `{ samples: Float32Array; sampleRate: number }` from
  `synthesize()`, NOT `AudioBuffer`.** The report asked for `AudioBuffer`.
  Reason: `AudioBuffer` is bound to a specific `AudioContext`, and if
  the consumer wants to feed it into THEIR own context (e.g.
  `MediaStreamAudioDestinationNode` for screen recording, or an offline
  render) they have to copy samples back out anyway. Raw floats are the
  universal interchange and let `play()` build the `AudioBuffer` lazily
  against its own context. If a consumer really needs the
  `AudioBuffer` we can add `synthesizeToAudioBuffer(text, opts,
  audioContext)` later.

- **`preloadWhenIdle` is a static factory, not an instance method.**
  Returns the instance. Otherwise the consumer would need to construct
  an instance first, defeating the fire-and-forget warm-cache use case.

- **Auto-warmup inside `init()`.** Yes — but expose `skipWarmup: true`
  for tests. Otherwise every test instance pays ~200ms of synthesis
  cost for no reason.

- **BCP-47 normalization in `_normalizeLanguage()` in `internal.js`.**
  Applies to both constructor input and `SpeakOptions.language`.
  `de-DE`/`de-AT`/`de-CH` all collapse to `de` for now; if we ever
  ship region-specific models, the registry key becomes the full tag.

- **Keep the side-effecting `import "@dittli/tts-de"` register call**
  alongside the explicit `{ dePack }` export. The report's
  "spooky action at a distance" complaint is real but blowing it up
  in the same release as the constructor reshape is more churn than
  the consumer wants. Both paths work; the side-effect path is the
  documented default ergonomic, the factory is the no-magic
  alternative.

- **Don't inline ORT WASMs as base64.** Re-export as static assets
  under `@dittli/tts-core/ort-wasm/*`. The consumer copy step is one
  line of build config; the inline alternative breaks browser caching
  and bloats the JS bundle for tabs that never call `speak()`.

- **`console.warn` monkey-patch only when `verbose: false`.** Globally
  patching `console.warn` from a library is normally a sin — be loud
  about it in the README and let `verbose: true` opt out.

## Out of scope

- **Test coverage.** Memory note `project_browser_testing_plan.md`
  lists four options. This reshape *moves the goalposts*:
  - Option 1 (fetch polyfill) becomes easier: no more hardcoded
    `new URL(..., import.meta.url)` in `g2p_en.js` / `g2p_predict.js`.
    The polyfill just needs `assetBase` to be a `file://` URL.
  - Option 2 (Vitest + happy-dom) becomes more attractive: with ORT
    centralized in `src/ort.js`, stubbing it is a one-line
    `vi.mock("@dittli/tts-core/internal")`.
  - Option 3 (`examples/browser-vite/` smoke app) becomes the natural
    home for the spike in step 1 — keep it after the spike as the
    canonical smoke check.
  - Option 4 (`vitest --browser`) is unchanged in difficulty.
- **Multi-voice / speaker switching** beyond the existing
  `metadata.spk2id` lookup.
- **Streaming synthesis** (no chunked output; whole utterance lands at
  once).
- **`init()` progress callback per-byte granularity.** The `onProgress`
  event is defined but Phase 1 fires it once per asset, not per-byte.
  Per-byte requires `ReadableStream` adapters around `fetch`; defer.
- **Node.js support.** v0.2.0 went browser-only; this reshape doesn't
  walk that back. `play()` is browser-only by definition; `synthesize()`
  *could* work in Node 22+ with a `file://` fetch polyfill, but that's
  not a supported configuration.

## Task tracking

Tasks are pre-created in the harness task list (see TaskList). 12 tasks:

1. Extract `tts-core/src/ort.js`
2. Extract `tts-core/src/audio.js`
3. Add `tts-core/src/internal.js`
4. Add `tts-core/src/engine.js`
5. Rewrite `tts-core/src/index.js`
6. Rewrite `tts-core/src/index.d.ts` + add `internal.d.ts`
7. Update `tts-core/package.json` + ORT WASM copy script
8. Relocate and rewrite `tts-de`
9. Relocate and rewrite `tts-en`
10. Update `scripts/check-publish-assets.js` manifest
11. Add `examples/browser-vite` smoke app
12. Update README with v0.4.0 consumer code

Resume by reading this doc, then `TaskList` to see status, then start
with task 1.
