import { DittliTTS } from "@dittli/tts-core";
import { graphemeToPhonemeDE } from "./g2p_de.js";

DittliTTS.registerLanguage("de", graphemeToPhonemeDE);
DittliTTS.registerDefaultMetadata("de", new URL("../metadata/dittli-de.json", import.meta.url));
DittliTTS.registerDefaultModel("de", new URL("../model/dittli-de_fp16.onnx", import.meta.url));

export { DittliTTS };
export default DittliTTS;
