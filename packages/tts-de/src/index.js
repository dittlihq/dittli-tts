const DittliTTS = require("@dittli/tts-core");
const { graphemeToPhonemeDE } = require("./g2p_de");

DittliTTS.registerLanguage("de", graphemeToPhonemeDE);

module.exports = DittliTTS;
