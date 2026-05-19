#!/usr/bin/env bash
# Pre-publish smoke test for `@dittli/tts-{core,de,en}`.
#
# Packs all three workspaces (runs `prepublishOnly`, respects each
# package's `files` allowlist), installs the tarballs into a throwaway
# consumer directory, copies assets into one tree exactly as a real
# consumer would, then synthesises a predefined sentence to a WAV file.
#
#   npm run test:publish              # English default
#   DITTLI_LANG=de npm run test:publish
#   DITTLI_TEXT="Custom sentence." npm run test:publish
#
# Listen to ./dist-pack/consumer/out.wav to confirm the package works.

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DIST="$ROOT/dist-pack"
CONSUMER="$DIST/consumer"

DITTLI_LANG="${DITTLI_LANG:-en}"
case "$DITTLI_LANG" in
  en) DEFAULT_TEXT="Hello, world. This is the published Dittli package, synthesizing speech from a tarball install." ;;
  de) DEFAULT_TEXT="Hallo Welt. Dies ist das veröffentlichte Dittli Paket, das Sprache aus einem Tarball-Install synthetisiert." ;;
  *)  DEFAULT_TEXT="Hello." ;;
esac
DITTLI_TEXT="${DITTLI_TEXT:-$DEFAULT_TEXT}"

echo "==> Resetting $DIST"
rm -rf "$DIST"
mkdir -p "$DIST"

echo "==> Packing all three packages (this runs prepublishOnly)"
npm pack --workspace=@dittli/tts-core --pack-destination="$DIST" --silent
npm pack --workspace=@dittli/tts-en   --pack-destination="$DIST" --silent
npm pack --workspace=@dittli/tts-de   --pack-destination="$DIST" --silent

echo
echo "==> Tarball contents (eyeball for surprises):"
for tgz in "$DIST"/*.tgz; do
  echo "--- $(basename "$tgz") ($(du -h "$tgz" | cut -f1)) ---"
  tar -tzf "$tgz" | sort
  echo
done

echo "==> Installing tarballs into $CONSUMER"
mkdir -p "$CONSUMER"
cat > "$CONSUMER/package.json" <<'JSON'
{
  "name": "dittli-pack-test",
  "version": "0.0.0",
  "private": true,
  "type": "module"
}
JSON

(
  cd "$CONSUMER"
  # All three at once so peerDependencies resolve in a single pass.
  npm install --no-package-lock --silent \
    "$DIST"/dittli-tts-core-*.tgz \
    "$DIST"/dittli-tts-en-*.tgz \
    "$DIST"/dittli-tts-de-*.tgz
)

echo "==> Copying assets into one tree (mirrors the consumer build step)"
rm -rf "$CONSUMER/assets"
mkdir -p "$CONSUMER/assets"
cp -r "$CONSUMER/node_modules/@dittli/tts-en/assets/en"   "$CONSUMER/assets/en"
cp -r "$CONSUMER/node_modules/@dittli/tts-de/assets/de"   "$CONSUMER/assets/de"
cp -r "$CONSUMER/node_modules/@dittli/tts-core/ort-wasm"  "$CONSUMER/assets/ort"

echo "==> Writing consumer test script"
cat > "$CONSUMER/run.mjs" <<'NODE'
/**
 * Synthesises a predefined sentence using the tarball-installed
 * packages and writes a WAV file to ./out.wav. Polyfills `fetch` for
 * `file://` URLs because Node's built-in fetch refuses them.
 */
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const realFetch = globalThis.fetch;
globalThis.fetch = async (url, opts) => {
  const s = String(url);
  if (!s.startsWith("file:")) return realFetch(url, opts);
  const path = fileURLToPath(s);
  const buf = await readFile(path);
  return {
    ok: true,
    status: 200,
    arrayBuffer: async () =>
      buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
    json: async () => JSON.parse(buf.toString("utf-8")),
  };
};

const lang = process.env.DITTLI_LANG || "en";
const text = process.env.DITTLI_TEXT || "Hello, world.";

const { DittliTTS, floatToWav } = await import("@dittli/tts-core");
await import(`@dittli/tts-${lang}`);

const assetBase = pathToFileURL(resolve("./assets") + "/").href;
console.log(`[run] language=${lang}`);
console.log(`[run] text="${text}"`);
console.log(`[run] assetBase=${assetBase}`);

const tts = new DittliTTS({ language: lang, assetBase, skipWarmup: true });
const { samples, sampleRate } = await tts.synthesize(text);
await tts.dispose();

const wav = floatToWav(samples, sampleRate);
const out = resolve("./out.wav");
await writeFile(out, wav);
console.log(
  `[run] wrote ${out} — ${wav.byteLength} bytes, ${samples.length} samples @ ${sampleRate} Hz`,
);
NODE

echo "==> Running synthesis"
(
  cd "$CONSUMER"
  DITTLI_LANG="$DITTLI_LANG" DITTLI_TEXT="$DITTLI_TEXT" node run.mjs
)

echo
echo "==> Done."
echo "    Play: $CONSUMER/out.wav"
