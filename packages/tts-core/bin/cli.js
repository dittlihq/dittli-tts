#!/usr/bin/env node

const DittliTTS = require("../src/index");
const { Command } = require("commander");

// Auto-register any language packs that are installed alongside this package.
for (const pkg of ["@dittli/tts-en", "@dittli/tts-de"]) {
  try {
    require(pkg);
  } catch (err) {
    if (err.code !== "MODULE_NOT_FOUND" || !err.message.includes(pkg)) {
      console.warn(`[dittli-tts] Warning: failed to load language pack ${pkg}:`, err.message);
    }
  }
}

const program = new Command();

program
  .name("dittli-tts")
  .description("Ultra-lightweight text-to-speech (1.6M params) — pure Node.js ONNX inference")
  .version(require("../package.json").version)
  .argument("<text>", "Text to synthesize")
  .option("-o, --output <path>", "Output file path", "output.wav")
  .option("-s, --speaker <id>", "Speaker ID", "MALE")
  .option("--speed <number>", "Speech speed", "1.0")
  .option("--model <path>", "Path to ONNX model file (required)")
  .option(
    "--metadata <path>",
    "Path to model metadata JSON (defaults to <model>.json or language-pack default)",
  )
  .option("--language <lang>", "Language hint when multiple packs are loaded (e.g. en, de)")
  .option("--device <dev>", "Device: cpu or gpu", "cpu")
  .action(async (text, options) => {
    try {
      const tts = new DittliTTS({
        modelPath: options.model,
        metadataPath: options.metadata,
        language: options.language,
        device: options.device,
      });

      await tts.speak(text, {
        output: options.output,
        speaker: options.speaker,
        speed: parseFloat(options.speed),
      });

      await tts.dispose();
    } catch (error) {
      console.error("Error:", error.message);
      process.exit(1);
    }
  });

program.parse();
