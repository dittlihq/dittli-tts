const path = require("node:path");
const DittliTTS = require("@dittli/tts-core");
const { graphemeToPhonemeDE } = require("./g2p_de");

DittliTTS.registerLanguage("de", graphemeToPhonemeDE);
DittliTTS.registerDefaultMetadata("de", path.join(__dirname, "../metadata/dittli-de.json"));

module.exports = DittliTTS;
