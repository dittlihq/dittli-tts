# browser-vite smoke

Canonical smoke check for the `@dittli/tts-*` v0.4.x browser API.

## Run

```bash
# from the repo root, once
npm install
node packages/tts-core/scripts/copy-ort-wasm.js

# then in this directory
npm run dev
```

Open http://localhost:5173/, pick a language, hit **Play**.

## What it demonstrates

- `new DittliTTS({ language, assetBase })` — no `import.meta.url`, no
  `modelUrl`/`metadataUrl`, no `LANG_PACKS` bookkeeping.
- `await tts.play(text)` — handles WAV → Blob → object-URL → `<audio>`
  internally.
- `AbortController` via `tts.stop()`.
- `onProgress` callback firing once per asset.

## How the assets get there

`copy-assets.mjs` (run as `predev` / `prebuild`) copies:

| From                                            | To                |
| ----------------------------------------------- | ----------------- |
| `packages/tts-en/assets/en/`                    | `public/tts/en/`  |
| `packages/tts-de/assets/de/`                    | `public/tts/de/`  |
| `packages/tts-core/ort-wasm/`                   | `public/tts/ort/` |

In a real consumer app this is one line of `vite.config.ts` (a
`viteStaticCopy` plugin or a `postinstall` hook).
