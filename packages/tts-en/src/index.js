const path = require("node:path");
const DittliTTS = require("@dittli/tts-core");
const { graphemeToPhonemeEN } = require("./g2p_en");

DittliTTS.registerLanguage("en", graphemeToPhonemeEN);

if (!DittliTTS._defaultMetadataPath) {
  DittliTTS._defaultMetadataPath = path.join(__dirname, "../metadata/dittli-en.json");
}

module.exports = DittliTTS;
