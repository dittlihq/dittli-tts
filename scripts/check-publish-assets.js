#!/usr/bin/env node
const fs = require("node:fs");
const path = require("node:path");

const MANIFEST = {
  "tts-core": ["src/index.js", "src/index.d.ts"],
  "tts-en": [
    "src/index.js",
    "src/g2p_en.js",
    "src/g2p_predict.js",
    "src/cmudict.json",
    "src/g2p_model.json",
    "metadata/dittli-en.json",
    "model/dittli-en_fp16.onnx",
  ],
  "tts-de": [
    "src/index.js",
    "src/g2p_de.js",
    "src/g2p_de_rules.json",
    "metadata/dittli-de.json",
    "model/dittli-de_fp16.onnx",
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
      `  python -m dittli_tts.inference.export --checkpoint checkpoints/G.pth --lang EN --out packages/tts-en/model/dittli-en.onnx\n` +
      `  python -m dittli_tts.inference.export --checkpoint checkpoints/G_de.pth --lang DE --out packages/tts-de/model/dittli-de.onnx`,
  );
  process.exit(1);
}

console.log(`[${pkg}] all ${MANIFEST[pkg].length} required files present.`);
