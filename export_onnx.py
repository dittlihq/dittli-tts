"""
Export TinyTTS to ONNX for maximum inference speed.
"""
import sys, os
import torch
sys.path.insert(0, r"c:\Users\VALTEC-07\Desktop\tiny-tts")
os.chdir(r"c:\Users\VALTEC-07\Desktop\tiny-tts")

from tiny_tts.infer import load_engine
from tiny_tts.utils.config import SPK2ID

print("Loading PyTorch model...")
model = load_engine("G.pth", device="cpu")
model.eval()

# Create dummy inputs for tracing
# Shape: (batch_size=1, seq_len=10)
seq_len = 10
x = torch.randint(0, 100, (1, seq_len), dtype=torch.long)
x_lengths = torch.LongTensor([seq_len])
sid = torch.LongTensor([SPK2ID["MALE"]])
tone = torch.randint(0, 5, (1, seq_len), dtype=torch.long)
language = torch.zeros((1, seq_len), dtype=torch.long)
bert = torch.zeros((1, 1024, seq_len), dtype=torch.float)
ja_bert = torch.zeros((1, 768, seq_len), dtype=torch.float)

# We need a wrapper module because ONNX export requires a single forward() method
class TinyTTS_ONNX_Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x, x_lengths, sid, tone, language, bert, ja_bert, noise_scale, noise_scale_w, length_scale):
        # We only return the audio output 'o', ignoring alignments and latents for inference speed
        o, _, _, _ = self.model.infer(
            x, x_lengths, sid, tone, language, bert, ja_bert,
            noise_scale=noise_scale, 
            noise_scale_w=noise_scale_w, 
            length_scale=length_scale
        )
        return o

print("Wrapping model for export...")
wrapper = TinyTTS_ONNX_Wrapper(model)
wrapper.eval()

# Scalar inputs need to be tensors for ONNX
noise_scale = torch.FloatTensor([0.667])
noise_scale_w = torch.FloatTensor([0.8])
length_scale = torch.FloatTensor([1.0])

dummy_inputs = (
    x, x_lengths, sid, tone, language, bert, ja_bert,
    noise_scale, noise_scale_w, length_scale
)

# Test forward pass
print("Testing forward pass before export...")
with torch.no_grad():
    out = wrapper(*dummy_inputs)
print(f"Output shape: {out.shape}")

print("Exporting to ONNX in FP32 (this may take a minute)...")
torch.onnx.export(
    wrapper,
    dummy_inputs,
    "tinytts.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=[
        "x", "x_lengths", "sid", "tone", "language", "bert", "ja_bert",
        "noise_scale", "noise_scale_w", "length_scale"
    ],
    output_names=["audio"],
    dynamic_axes={
        "x": {0: "batch_size", 1: "text_length"},
        "tone": {0: "batch_size", 1: "text_length"},
        "language": {0: "batch_size", 1: "text_length"},
        "bert": {0: "batch_size", 2: "text_length"},
        "ja_bert": {0: "batch_size", 2: "text_length"},
        "audio": {0: "batch_size", 2: "audio_length"}
    }
)

# Convert to FP16 for smaller size
try:
    from onnxruntime.quantization.shape_inference import quant_pre_process
    from onnxruntime.transformers.float16 import convert_float_to_float16
    from onnx import load_model, save_model
    
    print("Converting to FP16...")
    quant_pre_process("tinytts.onnx", "tinytts_infer.onnx", skip_symbolic_shape=True)
    model_fp32 = load_model("tinytts_infer.onnx")
    model_fp16 = convert_float_to_float16(model_fp32)
    save_model(model_fp16, "tinytts_fp16.onnx")
    
    fp32_size = os.path.getsize("tinytts.onnx") / (1024 * 1024)
    fp16_size = os.path.getsize("tinytts_fp16.onnx") / (1024 * 1024)
    print(f"ONNX FP32 Size: {fp32_size:.2f} MB")
    print(f"ONNX FP16 Size: {fp16_size:.2f} MB")
except ImportError:
    print("onnxruntime.quantization not available. Skipping FP16 conversion.")
    fp32_size = os.path.getsize("tinytts.onnx") / (1024 * 1024)
    print(f"ONNX FP32 Size: {fp32_size:.2f} MB")
