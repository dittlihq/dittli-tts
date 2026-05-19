import { registerLanguagePack } from "@dittli/tts-core/internal";
import { graphemeToPhonemeDE } from "./g2p_de.js";

export const dePack = {
  language: "de",
  g2p: graphemeToPhonemeDE,
  assets: {
    metadata: "de/metadata.json",
    model: "de/model.onnx",
  },
};

registerLanguagePack(dePack);

export { graphemeToPhonemeDE };
