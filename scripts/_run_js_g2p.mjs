/**
 * Reads a JSON array of words from stdin, prints { word: phones[] } JSON to
 * stdout. Used by tests/parity/test_g2p_de_parity.py — not a public entry point.
 *
 * Requires Node >= 22 for JSON modules with import attributes.
 */
import { graphemeToPhonemeDE } from "../packages/tts-de/src/g2p_de.js";

let buf = "";
process.stdin.setEncoding("utf-8");
process.stdin.on("data", (chunk) => {
  buf += chunk;
});
process.stdin.on("end", () => {
  const words = JSON.parse(buf);
  const out = {};
  for (const w of words) {
    const { phones } = graphemeToPhonemeDE(w, { padStartEnd: false });
    out[w] = phones;
  }
  process.stdout.write(JSON.stringify(out));
});
