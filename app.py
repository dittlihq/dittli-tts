import gradio as gr
import nltk
from tiny_tts import TinyTTS

# Download required NLTK data for g2p-en
try:
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('cmudict')
except Exception as e:
    print(f"NLTK download warning: {e}")

# Initialize the model (auto-downloads if needed)
print("Initializing TinyTTS...")
tts = TinyTTS()
print("Model loaded successfully!")

def synthesize_audio(text, speaker):
    output_path = "output.wav"
    try:
        tts.speak(text, output_path=output_path, speaker=speaker)
        return output_path
    except Exception as e:
        return f"Error: {e}"

# Define available speakers
# Define available speakers
available_speakers = list(tts.model.SPK2ID.keys()) if hasattr(tts.model, "SPK2ID") else ["female"]

# Create Gradio interface
with gr.Blocks(title="TinyTTS Demo", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🗣️ TinyTTS")
    gr.Markdown("**Ultra-lightweight English Text-to-Speech (~9M parameters, ~20MB on disk)**")
    gr.Markdown("This space runs on CPU efficiently and synthesizes high-quality audio faster than real-time.")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Input Text", 
                placeholder="Enter English text here...", 
                value="The weather is nice today, and I feel very relaxed.",
                lines=4
            )
            speaker_dropdown = gr.Dropdown(
                choices=available_speakers, 
                value="female" if "female" in available_speakers else available_speakers[0], 
                label="Speaker"
            )
            submit_btn = gr.Button("Synthesize Speech", variant="primary")
            
        with gr.Column():
            audio_output = gr.Audio(label="Output Audio", type="filepath")
            
    # Example prompts
    gr.Examples(
        examples=[
            ["The weather is nice today, and I feel very relaxed.", "female"],
            ["TinyTTS has only nine million parameters, making it extremely fast on CPUs.", "female"],
        ],
        inputs=[text_input, speaker_dropdown],
    )
    
    submit_btn.click(
        fn=synthesize_audio,
        inputs=[text_input, speaker_dropdown],
        outputs=audio_output
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0")
