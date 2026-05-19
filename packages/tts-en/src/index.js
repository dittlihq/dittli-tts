import { registerLanguagePack } from "@dittli/tts-core/internal";
import { graphemeToPhonemeEN } from "./g2p_en.js";

export const enPack = {
  language: "en",
  g2p: graphemeToPhonemeEN,
  assets: {
    metadata: "en/metadata.json",
    model: "en/model.onnx",
  },
};

registerLanguagePack(enPack);

export { graphemeToPhonemeEN };
