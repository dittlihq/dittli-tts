#!/usr/bin/env node

const TinyTTS = require("../index");
const { Command } = require("commander");

const program = new Command();

program
  .name("tiny-tts")
  .description("Ultra-lightweight text-to-speech (1.6M params) — pure Node.js ONNX inference")
  .version("4.0.0")
  .argument("<text>", "Text to synthesize")
  .option("-o, --output <path>", "Output file path", "output.wav")
  .option("-s, --speaker <id>", "Speaker ID", "MALE")
  .option("--speed <number>", "Speech speed", "1.0")
  .option("--model <path>", "Path to ONNX model (auto-downloaded if omitted)")
  .option(
    "--metadata <path>",
    "Path to model metadata JSON (defaults to <model>.json or English fallback)",
  )
  .option("--device <dev>", "Device: cpu or gpu", "cpu")
  .action(async (text, options) => {
    try {
      const tts = new TinyTTS({
        modelPath: options.model,
        metadataPath: options.metadata,
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
