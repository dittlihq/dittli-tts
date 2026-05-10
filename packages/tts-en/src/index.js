const path = require("node:path");
const DittliTTS = require("@dittli/tts-core");
const { graphemeToPhonemeEN } = require("./g2p_en");

DittliTTS.registerLanguage("en", graphemeToPhonemeEN);
DittliTTS.registerDefaultMetadata("en", path.join(__dirname, "../metadata/dittli-en.json"));
DittliTTS.registerDefaultModel("en", path.join(__dirname, "../model/dittli-en_fp16.onnx"));

module.exports = DittliTTS;
