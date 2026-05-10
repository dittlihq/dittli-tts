/**
 * DittliTTS — Pure Node.js text-to-speech via ONNX Runtime.
 *
 * This package is the inference engine only. Install a language pack to
 * register the G2P implementation for your target language:
 *
 *   require('@dittli/tts-en');  // English
 *   require('@dittli/tts-de');  // German
 *
 * Language packs call DittliTTS.registerLanguage() on load, so simply
 * requiring them is enough — no further setup needed.
 */

const ort = require("onnxruntime-node");
const { WaveFile } = require("wavefile");
const fs = require("node:fs");
const path = require("node:path");

const G2P_BY_LANG = {};
const DEFAULT_METADATA_BY_LANG = {};
const DEFAULT_MODEL_BY_LANG = {};

function _loadMetadata(metadataPath) {
  const raw = fs.readFileSync(metadataPath, "utf-8");
  const meta = JSON.parse(raw);
  for (const k of ["language", "language_id", "tone_offset", "sample_rate", "symbols"]) {
    if (meta[k] === undefined) {
      throw new Error(`metadata sidecar missing field "${k}": ${metadataPath}`);
    }
  }
  return meta;
}

function _phonemesToIds(phones, tones, meta) {
  const symbolSet = meta._symbolMap;
  const unkId = symbolSet.UNK;
  const phoneIds = phones.map((p) => (symbolSet[p] !== undefined ? symbolSet[p] : unkId));
  const toneIds = tones.map((t) => t + meta.tone_offset);
  const langIds = new Array(phoneIds.length).fill(meta.language_id);
  return [phoneIds, toneIds, langIds];
}

function _insertBlanks(arr) {
  const n = arr.length;
  const out = new Array(n * 2 + 1).fill(0);
  for (let i = 0; i < n; i++) out[1 + i * 2] = arr[i];
  return out;
}

class DittliTTS {
  constructor(opts = {}) {
    this.modelPath = opts.modelPath || null;
    this.metadataPath = opts.metadataPath || null;
    this.language = opts.language || null;
    this.device = opts.device || "cpu";
    this.session = null;
    this.metadata = null;
    this._symbolSet = null;
    this._initialized = false;
  }

  static registerLanguage(lang, g2pFn) {
    G2P_BY_LANG[lang] = g2pFn;
  }

  static registerDefaultMetadata(lang, metadataPath) {
    DEFAULT_METADATA_BY_LANG[lang] = metadataPath;
  }

  static registerDefaultModel(lang, modelPath) {
    DEFAULT_MODEL_BY_LANG[lang] = modelPath;
  }

  async init() {
    if (this._initialized) return;

    let modelPath = this.modelPath;
    if (!modelPath) {
      if (this.language && DEFAULT_MODEL_BY_LANG[this.language]
          && fs.existsSync(DEFAULT_MODEL_BY_LANG[this.language])) {
        modelPath = DEFAULT_MODEL_BY_LANG[this.language];
      } else {
        const registered = Object.keys(DEFAULT_MODEL_BY_LANG);
        if (registered.length === 1) {
          const p = DEFAULT_MODEL_BY_LANG[registered[0]];
          if (fs.existsSync(p)) modelPath = p;
        }
      }
    }
    if (!modelPath) {
      throw new Error(
        "No modelPath provided and no language pack supplied a default. " +
          "Install a language pack (e.g. @dittli/tts-en) or pass { modelPath }." +
          (Object.keys(DEFAULT_MODEL_BY_LANG).length > 1
            ? ` Loaded packs: ${Object.keys(DEFAULT_MODEL_BY_LANG).join(", ")} — pass { language: 'en' } to disambiguate.`
            : ""),
      );
    }
    if (!fs.existsSync(modelPath)) throw new Error(`Model not found: ${modelPath}`);

    let metadataPath = this.metadataPath;
    if (!metadataPath) {
      const guess = modelPath.replace(/\.onnx$/, ".json");
      if (fs.existsSync(guess)) {
        metadataPath = guess;
      } else if (this.language && DEFAULT_METADATA_BY_LANG[this.language]
                 && fs.existsSync(DEFAULT_METADATA_BY_LANG[this.language])) {
        metadataPath = DEFAULT_METADATA_BY_LANG[this.language];
      } else {
        const registered = Object.keys(DEFAULT_METADATA_BY_LANG);
        if (registered.length === 1) {
          const p = DEFAULT_METADATA_BY_LANG[registered[0]];
          if (fs.existsSync(p)) metadataPath = p;
        }
      }
      if (!metadataPath) {
        throw new Error(
          "No metadata sidecar found. Provide { metadataPath } or place a JSON file " +
            `next to ${modelPath} (same basename, .json extension).` +
            (Object.keys(DEFAULT_METADATA_BY_LANG).length > 1
              ? " Multiple language packs are loaded — pass { language: 'en' } to disambiguate."
              : ""),
        );
      }
    }

    const meta = _loadMetadata(metadataPath);
    const symMap = {};
    for (let i = 0; i < meta.symbols.length; i++) symMap[meta.symbols[i]] = i;
    meta._symbolMap = symMap;
    this._symbolSet = new Set(meta.symbols);
    this.metadata = meta;

    if (!G2P_BY_LANG[meta.language]) {
      throw new Error(
        `No G2P registered for language "${meta.language}". ` +
          `Install the matching language pack: @dittli/tts-${meta.language}`,
      );
    }

    console.log(`Loading ONNX model (${meta.language}, ${meta.symbols.length} symbols)...`);
    this.session = await ort.InferenceSession.create(modelPath, {
      executionProviders: [this.device === "gpu" ? "cuda" : "cpu"],
    });
    this._initialized = true;
    console.log(`Model loaded (${meta.sample_rate} Hz).`);
  }

  textToPhonemeIds(text) {
    const g2p = G2P_BY_LANG[this.metadata.language];
    const { phones, tones } = g2p(text, { symbolSet: this._symbolSet });
    const [phoneIds, toneIds, langIds] = _phonemesToIds(phones, tones, this.metadata);
    return {
      phoneIds: _insertBlanks(phoneIds),
      toneIds: _insertBlanks(toneIds),
      langIds: _insertBlanks(langIds),
    };
  }

  async speak(text, options = {}) {
    await this.init();

    let outputPath = "output.wav";
    let speakerName = null;
    let speed = 1.0;
    if (typeof options === "string") {
      outputPath = options;
    } else {
      outputPath = options.output || "output.wav";
      speakerName = options.speaker || null;
      speed = options.speed || 1.0;
    }

    const meta = this.metadata;
    const spk2id = meta.spk2id || {};
    let sidValue = 0;
    if (speakerName && spk2id[speakerName] !== undefined) {
      sidValue = spk2id[speakerName];
    } else if (speakerName) {
      console.warn(
        `[DittliTTS] Unknown speaker "${speakerName}", using ID 0. ` +
          `Known: ${Object.keys(spk2id).join(", ") || "(none)"}`,
      );
    }

    console.log("Synthesizing:", text);
    const { phoneIds, toneIds, langIds } = this.textToPhonemeIds(text);
    const seqLen = phoneIds.length;
    console.log("Phonemes:", seqLen, "tokens");

    const feeds = {
      x: new ort.Tensor("int64", BigInt64Array.from(phoneIds.map((v) => BigInt(v))), [1, seqLen]),
      x_lengths: new ort.Tensor("int64", [BigInt(seqLen)], [1]),
      sid: new ort.Tensor("int64", [BigInt(sidValue)], [1]),
      tone: new ort.Tensor("int64", BigInt64Array.from(toneIds.map((v) => BigInt(v))), [1, seqLen]),
      language: new ort.Tensor("int64", BigInt64Array.from(langIds.map((v) => BigInt(v))), [
        1,
        seqLen,
      ]),
      bert: new ort.Tensor("float32", new Float32Array(seqLen * 1024), [1, 1024, seqLen]),
      ja_bert: new ort.Tensor("float32", new Float32Array(seqLen * 768), [1, 768, seqLen]),
      noise_scale: new ort.Tensor("float32", [0.667], [1]),
      noise_scale_w: new ort.Tensor("float32", [0.8], [1]),
      length_scale: new ort.Tensor("float32", [1.0 / speed], [1]),
    };

    const results = await this.session.run(feeds);
    const audio = results.audio.data;

    const wav = new WaveFile();
    wav.fromScratch(1, meta.sample_rate, "32f", Array.from(audio));
    const wavBuf = wav.toBuffer();

    const outDir = path.dirname(outputPath);
    if (outDir && outDir !== "." && !fs.existsSync(outDir))
      fs.mkdirSync(outDir, { recursive: true });
    fs.writeFileSync(outputPath, wavBuf);
    console.log("Saved:", outputPath, `(${(audio.length / meta.sample_rate).toFixed(2)}s)`);
    return wavBuf;
  }

  async dispose() {
    if (this.session) {
      this.session.release();
      this.session = null;
      this._initialized = false;
    }
  }
}

module.exports = DittliTTS;
