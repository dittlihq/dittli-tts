#!/usr/bin/env node
/**
 * Copy the subset of `onnxruntime-web/dist/` WASMs that we re-export
 * under `@dittli/tts-core/ort-wasm/`. Runs as `prepublishOnly`.
 *
 * Consumers copy `node_modules/@dittli/tts-core/ort-wasm/` into their
 * static asset tree alongside their per-language assets.
 */

import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));

const ASSETS = [
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm",
  "ort-wasm-simd-threaded.jsep.mjs",
  "ort-wasm-simd-threaded.jsep.wasm",
];

function findOrtDist(start) {
  let dir = start;
  for (let i = 0; i < 6; i++) {
    const cand = join(dir, "node_modules", "onnxruntime-web", "dist");
    if (existsSync(cand)) return cand;
    const parent = dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return null;
}

const pkgRoot = resolve(HERE, "..");
const dist = findOrtDist(pkgRoot);
if (!dist) {
  console.error("copy-ort-wasm: could not locate onnxruntime-web/dist — run `npm install` first");
  process.exit(1);
}
const outDir = join(pkgRoot, "ort-wasm");
mkdirSync(outDir, { recursive: true });

let copied = 0;
for (const asset of ASSETS) {
  const src = join(dist, asset);
  if (!existsSync(src)) {
    console.error(`copy-ort-wasm: missing source asset ${src}`);
    process.exit(1);
  }
  copyFileSync(src, join(outDir, asset));
  copied++;
}
console.log(`copy-ort-wasm: copied ${copied} asset(s) → ${outDir}`);
