/**
 * Parity harness: run the *committed* English G2P ONNX graphs through the real
 * `createOnnxG2p` host loop (the same code the browser uses) via onnxruntime-node,
 * and assert the output matches `g2p_en` — the library the weights came from.
 *
 * This is the browser-path proxy for `scripts/export_g2p_onnx.py --verify`
 * (which checks the graphs via Python onnxruntime). Not in CI; run on demand:
 *
 *   npm i --no-save onnxruntime-node && node scripts/_g2p_onnx_parity.mjs
 *
 * The reference phonemes below are g2p_en's own output, captured from the
 * verified Python run, so matching them is parity with g2p_en.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import * as ort from "onnxruntime-node";

import { createOnnxG2p } from "../packages/tts-core/src/g2p_onnx.js";

const ASSETS = join(
  dirname(fileURLToPath(import.meta.url)),
  "..",
  "packages",
  "tts-en",
  "assets",
  "en",
);
const bytes = (f) => new Uint8Array(readFileSync(join(ASSETS, f)));

const REFERENCE = {
  blorptang: ["B", "L", "AO1", "R", "P", "T", "AE2", "NG"],
  zylophonics: ["Z", "AY2", "L", "OW0", "F", "AA1", "N", "IH0", "K", "S"],
  qwertzuiop: ["K", "W", "ER1", "T", "S", "UW2", "P"],
  schmenkle: ["SH", "M", "EH1", "NG", "K", "AH0", "L"],
  kubernetes: ["K", "AH0", "B", "ER1", "N", "AH0", "T", "S"],
  tensorflow: ["T", "EH1", "N", "S", "ER0", "L", "OW0", "F"],
  dittli: ["D", "IH1", "T", "L", "IY0"],
  vauxen: ["V", "AO1", "K", "S", "AH0", "N"],
  glorptastic: ["G", "L", "AO2", "R", "P", "T", "AE1", "S", "T", "IH0", "K"],
};

const ortShim = {
  createSession: (b) => ort.InferenceSession.create(b),
  runSession: (session, feeds) => session.run(feeds),
  tensor: (type, data, shape) => new ort.Tensor(type, data, shape),
};

const predict = await createOnnxG2p({
  ort: ortShim,
  encoderBytes: bytes("g2p_encoder.onnx"),
  decoderBytes: bytes("g2p_decoder_step.onnx"),
  vocab: JSON.parse(readFileSync(join(ASSETS, "g2p_vocab.json"), "utf-8")),
});

let mismatches = 0;
for (const [word, ref] of Object.entries(REFERENCE)) {
  const got = await predict(word);
  const ok = JSON.stringify(got) === JSON.stringify(ref);
  if (!ok) mismatches++;
  console.log(`  [${ok ? "ok" : "MISMATCH"}] ${word.padEnd(14)} ${JSON.stringify(got)}`);
}
const n = Object.keys(REFERENCE).length;
console.log(`\n${n - mismatches}/${n} words match g2p_en (via the real createOnnxG2p host loop).`);
process.exit(mismatches ? 1 : 0);
