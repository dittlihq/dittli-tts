#!/usr/bin/env node
/**
 * Copy the wasm-pack output from `packages/tts-runtime/pkg/` into
 * `packages/tts-core/runtime-wasm/`. Runs as `prepublishOnly`.
 *
 * Consumers copy `node_modules/@dittli/tts-core/runtime-wasm/` into their
 * static asset tree. The JS glue (`dittli_runtime.js`) is bundled with the
 * package; only `dittli_runtime_bg.wasm` is fetched at runtime from the
 * consumer's `runtimeAssetBase` URL.
 */

import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const pkgRoot = resolve(HERE, "..");
const runtimePkg = resolve(pkgRoot, "../../packages/tts-runtime/pkg");

const ASSETS = ["dittli_runtime.js", "dittli_runtime.d.ts", "dittli_runtime_bg.wasm"];

if (!existsSync(runtimePkg)) {
  console.error(
    "copy-runtime-wasm: packages/tts-runtime/pkg/ not found.\n" +
      "Run: cd packages/tts-runtime && wasm-pack build --target web --release",
  );
  process.exit(1);
}

const outDir = join(pkgRoot, "runtime-wasm");
mkdirSync(outDir, { recursive: true });

let copied = 0;
for (const asset of ASSETS) {
  const src = join(runtimePkg, asset);
  if (!existsSync(src)) {
    console.error(`copy-runtime-wasm: missing source asset ${src}`);
    process.exit(1);
  }
  copyFileSync(src, join(outDir, asset));
  copied++;
}
console.log(`copy-runtime-wasm: copied ${copied} asset(s) → ${outDir}`);
