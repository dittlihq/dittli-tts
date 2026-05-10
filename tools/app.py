import os

import gradio as gr
import nltk

from dittli_tts import DittliTTS

# Download required NLTK data for g2p-en
try:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('cmudict', quiet=True)
except Exception as e:
    print(f"NLTK download warning: {e}")

_checkpoint = os.environ.get("DITTLI_CHECKPOINT", "checkpoints/G.pth")
print(f"Initializing DittliTTS from {_checkpoint} ...")
tts = DittliTTS(checkpoint_path=_checkpoint)
print("Model loaded successfully!")


def synthesize_audio(text, speed):
    output_path = "output.wav"
    try:
        tts.speak(text, output_path=output_path, speaker="MALE", speed=speed)
        return output_path
    except Exception as e:
        return f"Error: {e}"


COMPARISON_TABLE = """
## ⚡ Comparison with Other TTS Engines

All numbers are **CPU-only** on the same Intel Core laptop. Text: *"The weather is nice today, and I feel very relaxed."*

| ENGINE | Params | Total (s) | Audio (s) | RTFx |
|:---|---:|---:|---:|---:|
| **DittliTTS (ONNX) 🚀** | **1.6M** | **0.092** | **4.88** | **~53x** |
| Piper (ONNX, 22kHz) | ~63M | 0.112 | 2.91 | ~26x |
| DittliTTS (PyTorch) | 1.6M | 0.272 | 4.88 | ~18x |
| KittenTTS nano | ~10M | 0.286 | 4.87 | ~17x |
| Supertonic (2-step) | ~82M | 0.249 | 3.69 | ~15x |
| Pocket-TTS | 100M | 0.928 | 3.68 | ~4x |
| Kokoro ONNX | 82M | 0.933 | 3.16 | ~3x |
| KittenTTS mini | ~25M | 2.047 | 4.17 | ~2x |

> **RTFx** = Audio Duration ÷ Synthesis Time (higher = faster).
> DittliTTS achieves the **best speed-to-size ratio**: only **1.6M params** / **3.4 MB** ONNX yet ~53× real-time at 44.1kHz.
"""

# Create Gradio interface
with gr.Blocks(title="DittliTTS Demo", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🗣️ DittliTTS")
    gr.Markdown(
        "**Ultra-lightweight English Text-to-Speech — only 1.6M parameters, ~3.4 MB ONNX**\n\n"
        "This space runs on CPU efficiently and synthesizes high-quality 44.1kHz audio **~53× faster** than real-time."
    )

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Enter English text here...",
                value="The weather is nice today, and I feel very relaxed.",
                lines=4
            )
            speed_slider = gr.Slider(
                minimum=0.5,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="Speed (1.0 = normal, >1 = faster, <1 = slower)"
            )
            submit_btn = gr.Button("🔊 Synthesize Speech", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio", type="filepath")

    # Example prompts
    gr.Examples(
        examples=[
            ["The weather is nice today, and I feel very relaxed.", 1.0],
            ["DittliTTS has only one point six million parameters, making it extremely fast on CPUs.", 1.0],
            ["This is a speed test. Speaking at one and a half times the normal rate.", 1.5],
            ["Slow and steady wins the race. Let me speak more carefully.", 0.7],
        ],
        inputs=[text_input, speed_slider],
    )

    submit_btn.click(
        fn=synthesize_audio,
        inputs=[text_input, speed_slider],
        outputs=audio_output
    )

    # Comparison table
    gr.Markdown(COMPARISON_TABLE)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0")
