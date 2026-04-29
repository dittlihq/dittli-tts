/**
 * English grapheme-to-phoneme — extracted from the original index.js.
 * Mirrors tiny_tts/text/english.py: CMU dictionary lookup with a neural G2P
 * fallback (g2p_predict.js) for OOV words.
 */
const fs = require("node:fs");
const path = require("node:path");
const g2pPredict = require("./g2p_predict");

let CMU = {};
const _cmuDictPath = path.join(__dirname, "cmudict.json");
if (fs.existsSync(_cmuDictPath)) {
  try {
    CMU = JSON.parse(fs.readFileSync(_cmuDictPath, "utf-8"));
  } catch (e) {
    console.warn("[TinyTTS] Failed to load cmudict.json:", e.message);
  }
}
if (Object.keys(CMU).length === 0) {
  console.warn(
    "[TinyTTS] cmudict.json missing; English G2P will lean heavily on the neural fallback.",
  );
}

function _parsePhone(phn) {
  const m = phn.match(/(\d)$/);
  if (m) return [phn.slice(0, -1).toLowerCase(), parseInt(m[1], 10) + 1];
  return [phn.toLowerCase(), 0];
}

function _parseSyllables(syllables) {
  const phones = [];
  const tones = [];
  for (const syl of syllables) {
    for (const phn of syl) {
      const [ph, tone] = _parsePhone(phn);
      phones.push(ph);
      tones.push(tone);
    }
  }
  return [phones, tones];
}

function _mapPhoneme(ph, symbolSet) {
  const rep = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "\xB7": ",",
    "、": ",",
    "...": "…",
    v: "V",
  };
  if (rep[ph] !== undefined) ph = rep[ph];
  if (symbolSet && !symbolSet.has(ph)) return "UNK";
  return ph;
}

function graphemeToPhonemeEN(text, opts = {}) {
  const { symbolSet = null, padStartEnd = true } = opts;
  text = text.toLowerCase().trim();
  const words = text.split(/\s+/).filter((w) => w.length > 0);
  const allPhones = [];
  const allTones = [];
  const word2ph = [];

  for (const word of words) {
    const lead = (word.match(/^[^a-z0-9]*/) || [""])[0];
    const trail = (word.match(/[^a-z0-9']*$/) || [""])[0];
    const core = word.slice(lead.length, word.length - trail.length);

    for (const ch of lead) {
      allPhones.push(_mapPhoneme(ch, symbolSet));
      allTones.push(0);
      word2ph.push(1);
    }

    if (core.length > 0) {
      let resolved = false;

      if (core.includes("'")) {
        const parts = core.split("'");
        const partPhones = [];
        const partTones = [];
        let allFound = true;
        for (let pi = 0; pi < parts.length; pi++) {
          const part = parts[pi];
          if (pi > 0) {
            partPhones.push("'");
            partTones.push(0);
          }
          if (part.length === 0) continue;
          const upper = part.toUpperCase();
          if (CMU[upper]) {
            const [ph, tn] = _parseSyllables([CMU[upper]]);
            partPhones.push(...ph);
            partTones.push(...tn);
          } else {
            const preds = g2pPredict.predict(part);
            if (preds && preds.length > 0) {
              for (const phn of preds) {
                const [ph2, tn2] = _parsePhone(phn);
                partPhones.push(ph2);
                partTones.push(tn2);
              }
            } else {
              allFound = false;
              break;
            }
          }
        }
        if (allFound && partPhones.length > 0) {
          for (const p of partPhones) allPhones.push(_mapPhoneme(p, symbolSet));
          allTones.push(...partTones);
          word2ph.push(partPhones.length);
          resolved = true;
        }
      }

      if (!resolved) {
        const upper = core.toUpperCase();
        if (CMU[upper]) {
          const [phones, tones] = _parseSyllables([CMU[upper]]);
          for (const p of phones) allPhones.push(_mapPhoneme(p, symbolSet));
          allTones.push(...tones);
          word2ph.push(phones.length);
          resolved = true;
        }
      }

      if (!resolved) {
        const preds = g2pPredict.predict(core);
        if (preds && preds.length > 0) {
          const [ph, tn] = _parseSyllables([preds]);
          for (const p of ph) allPhones.push(_mapPhoneme(p, symbolSet));
          allTones.push(...tn);
          word2ph.push(ph.length);
        } else {
          for (const ch of core) {
            if (ch === "'") continue;
            allPhones.push(_mapPhoneme(ch.toLowerCase(), symbolSet));
            allTones.push(0);
          }
          word2ph.push(core.replace(/'/g, "").length);
        }
      }
    }

    for (const ch of trail) {
      allPhones.push(_mapPhoneme(ch, symbolSet));
      allTones.push(0);
      word2ph.push(1);
    }
  }

  if (padStartEnd) {
    allPhones.unshift("_");
    allPhones.push("_");
    allTones.unshift(0);
    allTones.push(0);
    word2ph.unshift(1);
    word2ph.push(1);
  }
  return { phones: allPhones, tones: allTones, word2ph };
}

module.exports = { graphemeToPhonemeEN };
