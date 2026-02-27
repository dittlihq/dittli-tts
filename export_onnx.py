"""
Export TinyTTS model components to ONNX format.

Exports 4 ONNX sub-graphs:
  1. text_encoder.onnx   - phoneme IDs → hidden states
  2. duration_predictor.onnx - hidden states → durations
  3. flow.onnx           - latent flow (reverse)
  4. decoder.onnx        - latent → audio waveform
"""
import os
import sys
import torch
import torch.nn as nn
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from tiny_tts.infer import load_engine
from tiny_tts.utils.config import (
    SPEC_CHANNELS, SEGMENT_FRAMES, N_SPEAKERS, MODEL_PARAMS,
    SPK2ID
)
from tiny_tts.text.symbols import symbols
from tiny_tts.utils.config import MODEL_PARAMS


# ─────────────────────────────────────────────
# ONNX Wrapper Modules
# ─────────────────────────────────────────────

class TextEncoderONNX(nn.Module):
    """enc_p + emb_g → (x_enc, m_p, logs_p, x_mask, g)"""
    def __init__(self, model):
        super().__init__()
        self.enc_p = model.enc_p
        self.emb_g = model.emb_g
        self.use_vc = model.use_vc

    def forward(self, x, x_lengths, tone, language, bert, ja_bert, sid):
        g = self.emb_g(sid).unsqueeze(-1)  # [B, gin, 1]
        g_p = None if self.use_vc else g
        x_enc, m_p, logs_p, x_mask = self.enc_p(
            x, x_lengths, tone, language, bert, ja_bert, g=g_p
        )
        return x_enc, m_p, logs_p, x_mask, g


class DurationPredictorONNX(nn.Module):
    """dp → logw"""
    def __init__(self, model):
        super().__init__()
        self.dp = model.dp

    def forward(self, x, x_mask, g):
        return self.dp(x, x_mask, g=g)


class FlowONNX(nn.Module):
    """flow (reverse) → z"""
    def __init__(self, model):
        super().__init__()
        self.flow = model.flow

    def forward(self, z_p, y_mask, g):
        return self.flow(z_p, y_mask, g=g, reverse=True)


class DecoderONNX(nn.Module):
    """dec → audio"""
    def __init__(self, model):
        super().__init__()
        self.dec = model.dec

    def forward(self, z, g):
        return self.dec(z, g=g)


# ─────────────────────────────────────────────
# Export Functions
# ─────────────────────────────────────────────

def export_text_encoder(model, output_dir: Path, hidden=80, inter=80):
    print("\n[1/4] Exporting Text Encoder...")
    enc = TextEncoderONNX(model).eval()
    B, S = 1, 50
    dummy = (
        torch.randint(0, len(symbols), (B, S)),            # phone_ids
        torch.LongTensor([S]),                              # phone_lengths
        torch.randint(0, 16, (B, S)),                       # tone
        torch.zeros(B, S, dtype=torch.long),                # language
        torch.zeros(B, 1024, S),                            # bert
        torch.zeros(B, 768, S),                             # ja_bert
        torch.LongTensor([0]),                              # sid
    )
    out_path = output_dir / "text_encoder.onnx"
    try:
        torch.onnx.export(
            enc, dummy, str(out_path),
            input_names=["phone_ids", "phone_lengths", "tone_ids",
                         "language_ids", "bert", "ja_bert", "speaker_id"],
            output_names=["x_enc", "m_p", "logs_p", "x_mask", "g"],
            dynamic_axes={
                "phone_ids":     {0: "B", 1: "T"},
                "phone_lengths": {0: "B"},
                "tone_ids":      {0: "B", 1: "T"},
                "language_ids":  {0: "B", 1: "T"},
                "bert":          {0: "B", 2: "T"},
                "ja_bert":       {0: "B", 2: "T"},
                "speaker_id":    {0: "B"},
                "x_enc":         {0: "B", 2: "T"},
                "m_p":           {0: "B", 2: "T"},
                "logs_p":        {0: "B", 2: "T"},
                "x_mask":        {0: "B", 2: "T"},
                "g":             {0: "B"},
            },
            opset_version=14,
            do_constant_folding=True,
        )
        mb = out_path.stat().st_size / 1024**2
        print(f"  ✅ Saved: {out_path} ({mb:.2f} MB)")
        return out_path
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return None


def export_duration_predictor(model, output_dir: Path, hidden=80, gin=80):
    print("\n[2/4] Exporting Duration Predictor...")
    dp = DurationPredictorONNX(model).eval()
    B, T = 1, 50
    dummy = (
        torch.randn(B, hidden, T),   # x
        torch.ones(B, 1, T),         # x_mask
        torch.randn(B, gin, 1),      # g
    )
    out_path = output_dir / "duration_predictor.onnx"
    try:
        torch.onnx.export(
            dp, dummy, str(out_path),
            input_names=["x", "x_mask", "g"],
            output_names=["logw"],
            dynamic_axes={
                "x":      {0: "B", 2: "T"},
                "x_mask": {0: "B", 2: "T"},
                "g":      {0: "B"},
                "logw":   {0: "B", 2: "T"},
            },
            opset_version=14,
            do_constant_folding=True,
        )
        mb = out_path.stat().st_size / 1024**2
        print(f"  ✅ Saved: {out_path} ({mb:.2f} MB)")
        return out_path
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return None


def export_flow(model, output_dir: Path, inter=80, gin=80):
    print("\n[3/4] Exporting Flow (reverse)...")
    flow = FlowONNX(model).eval()
    B, T = 1, 100
    dummy = (
        torch.randn(B, inter, T),    # z_p
        torch.ones(B, 1, T),         # y_mask
        torch.randn(B, gin, 1),      # g
    )
    out_path = output_dir / "flow.onnx"
    try:
        torch.onnx.export(
            flow, dummy, str(out_path),
            input_names=["z_p", "y_mask", "g"],
            output_names=["z"],
            dynamic_axes={
                "z_p":   {0: "B", 2: "T"},
                "y_mask":{0: "B", 2: "T"},
                "g":     {0: "B"},
                "z":     {0: "B", 2: "T"},
            },
            opset_version=14,
            do_constant_folding=True,
        )
        mb = out_path.stat().st_size / 1024**2
        print(f"  ✅ Saved: {out_path} ({mb:.2f} MB)")
        return out_path
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return None


def export_decoder(model, output_dir: Path, inter=80, gin=80):
    print("\n[4/4] Exporting Decoder...")
    dec = DecoderONNX(model).eval()
    B, T = 1, 100
    dummy = (
        torch.randn(B, inter, T),    # z
        torch.randn(B, gin, 1),      # g
    )
    out_path = output_dir / "decoder.onnx"
    try:
        torch.onnx.export(
            dec, dummy, str(out_path),
            input_names=["z", "g"],
            output_names=["audio"],
            dynamic_axes={
                "z":     {0: "B", 2: "T"},
                "g":     {0: "B"},
                "audio": {0: "B", 2: "samples"},
            },
            opset_version=14,
            do_constant_folding=True,
        )
        mb = out_path.stat().st_size / 1024**2
        print(f"  ✅ Saved: {out_path} ({mb:.2f} MB)")
        return out_path
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return None


def verify_onnx(path: Path):
    import onnxruntime as ort
    try:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        ins  = [i.name for i in sess.get_inputs()]
        outs = [o.name for o in sess.get_outputs()]
        print(f"    inputs : {ins}")
        print(f"    outputs: {outs}")
        return True
    except Exception as e:
        print(f"    ⚠️  verify failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export TinyTTS to ONNX")
    parser.add_argument("--checkpoint", "-c", default="checkpoints/G.pth",
                        help="Path to G.pth checkpoint")
    parser.add_argument("--output-dir", "-o", default="onnx",
                        help="Directory to save ONNX files")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"Loading model from {args.checkpoint} …")
    model = load_engine(args.checkpoint, device="cpu")
    model.eval()
    print("Model loaded.\n")

    # Read actual dims from config
    hidden = MODEL_PARAMS["hidden_channels"]   # 80
    inter  = MODEL_PARAMS["inter_channels"]    # 80
    gin    = MODEL_PARAMS["gin_channels"]      # 80

    results = [
        export_text_encoder    (model, out_dir, hidden, inter),
        export_duration_predictor(model, out_dir, hidden, gin),
        export_flow            (model, out_dir, inter, gin),
        export_decoder         (model, out_dir, inter, gin),
    ]

    print("\n\n=== Export Summary ===")
    names = ["Text Encoder", "Duration Predictor", "Flow", "Decoder"]
    total = 0
    for name, path in zip(names, results):
        if path:
            mb = path.stat().st_size / 1024**2
            total += mb
            print(f"  ✅ {name}: {path.name}  ({mb:.2f} MB)")
            verify_onnx(path)
        else:
            print(f"  ❌ {name}: FAILED")
    print(f"\n  Total ONNX size: {total:.2f} MB")


if __name__ == "__main__":
    main()
