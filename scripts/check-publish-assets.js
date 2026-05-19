#!/usr/bin/env node
const fs = require("node:fs");
const path = require("node:path");

const ORT_WASM_ASSETS = [
  "ort-wasm/ort-wasm-simd-threaded.mjs",
  "ort-wasm/ort-wasm-simd-threaded.wasm",
];

const MANIFEST = {
  "tts-core": [
    "src/index.js",
    "src/index.d.ts",
    "src/internal.js",
    "src/internal.d.ts",
    "src/ort.js",
    "src/audio.js",
    "src/engine.js",
    ...ORT_WASM_ASSETS,
  ],
  "tts-en": [
    "src/index.js",
    "src/index.d.ts",
    "src/g2p_en.js",
    "src/g2p_predict.js",
    "assets/en/metadata.json",
    "assets/en/model.onnx",
    "assets/en/cmudict.json",
    "assets/en/g2p_model.json",
  ],
  "tts-de": [
    "src/index.js",
    "src/index.d.ts",
    "src/g2p_de.js",
    "src/g2p_de_rules.json",
    "assets/de/metadata.json",
    "assets/de/model.onnx",
  ],
};

const pkg = process.argv[2];
if (!pkg || !MANIFEST[pkg]) {
  console.error(`Usage: check-publish-assets.js <${Object.keys(MANIFEST).join("|")}>`);
  process.exit(2);
}

const pkgRoot = path.resolve(__dirname, "..", "packages", pkg);
const missing = MANIFEST[pkg].filter((rel) => !fs.existsSync(path.join(pkgRoot, rel)));

if (missing.length) {
  console.error(`[${pkg}] cannot publish — ${missing.length} required file(s) missing:`);
  for (const m of missing) console.error(`  - ${m}`);
  console.error(
    `\nIf an ONNX model is missing, re-export it:\n` +
      `  python -m dittli_tts.inference.export --checkpoint checkpoints/G.pth --lang EN --out packages/tts-en/assets/en/model.onnx\n` +
      `  python -m dittli_tts.inference.export --checkpoint checkpoints/G_de.pth --lang DE --out packages/tts-de/assets/de/model.onnx\n` +
      `\nIf an ORT WASM asset is missing in tts-core, run:\n` +
      `  node packages/tts-core/scripts/copy-ort-wasm.js`,
  );
  process.exit(1);
}

console.log(`[${pkg}] all ${MANIFEST[pkg].length} required files present.`);
