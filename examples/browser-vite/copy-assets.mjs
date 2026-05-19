/**
 * Mirror what a real consumer's build config does: copy the per-language
 * assets and the re-exported ORT WASMs into `public/tts/` so Vite serves
 * them at `/tts/...`.
 *
 * In a normal consumer app this is one line in `vite.config.ts` (or a
 * `postinstall` script). We do it explicitly here to keep the smoke
 * app vanilla.
 */

import { cpSync, existsSync, mkdirSync, rmSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "..", "..");

const DEST = resolve(HERE, "public", "tts");
rmSync(DEST, { recursive: true, force: true });
mkdirSync(DEST, { recursive: true });

const COPIES = [
  { from: resolve(ROOT, "packages/tts-en/assets/en"), to: resolve(DEST, "en") },
  { from: resolve(ROOT, "packages/tts-de/assets/de"), to: resolve(DEST, "de") },
  { from: resolve(ROOT, "packages/tts-core/ort-wasm"), to: resolve(DEST, "ort") },
];

for (const { from, to } of COPIES) {
  if (!existsSync(from)) {
    if (from.endsWith("ort-wasm")) {
      console.error(
        `[copy-assets] missing ${from}.\nRun: node packages/tts-core/scripts/copy-ort-wasm.js`,
      );
      process.exit(1);
    }
    console.error(`[copy-assets] missing ${from}`);
    process.exit(1);
  }
  cpSync(from, to, { recursive: true });
  console.log(`[copy-assets] ${from} → ${to}`);
}
