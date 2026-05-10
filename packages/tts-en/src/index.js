const path = require("node:path");
const DittliTTS = require("@dittli/tts-core");
const { graphemeToPhonemeEN } = require("./g2p_en");

DittliTTS.registerLanguage("en", graphemeToPhonemeEN);
DittliTTS.registerDefaultMetadata("en", path.join(__dirname, "../metadata/dittli-en.json"));

module.exports = DittliTTS;
