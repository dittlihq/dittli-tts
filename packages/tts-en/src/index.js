import { DittliTTS } from "@dittli/tts-core";
import { graphemeToPhonemeEN } from "./g2p_en.js";

DittliTTS.registerLanguage("en", graphemeToPhonemeEN);
DittliTTS.registerDefaultMetadata("en", new URL("../metadata/dittli-en.json", import.meta.url));
DittliTTS.registerDefaultModel("en", new URL("../model/dittli-en_fp16.onnx", import.meta.url));

export { DittliTTS };
export default DittliTTS;
