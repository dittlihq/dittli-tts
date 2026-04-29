"""Export DittliTTS to ONNX and write the model metadata sidecar JSON.

Run from the repo root:
    python export_onnx.py --checkpoint checkpoints/G.pth \
        --out models/dittli.onnx --lang EN
"""
import argparse
import json
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from dittli_tts.infer import load_engine
from dittli_tts.text.symbols import (
    language_id_map,
    language_tone_start_map,
)
from dittli_tts.text.symbols import (
    symbols as new_symbols,
)
from dittli_tts.utils.config import SAMPLING_RATE


def _resolve_symbols(lang: str) -> list[str]:
    """English ONNX was trained against the pre-German 219-symbol list. If a
    snapshot exists in checkpoints/symbols_v1_en.txt and the model's vocab
    matches its length, prefer it. Otherwise use the current symbol table."""
    if lang.upper() == "EN":
        snap = os.path.join(REPO_ROOT, "checkpoints", "symbols_v1_en.txt")
        if os.path.exists(snap):
            with open(snap, encoding="utf-8") as f:
                return [line.rstrip("\n") for line in f]
    return list(new_symbols)


def _build_metadata(lang: str, spk2id: dict[str, int], symbols: list[str]) -> dict:
    return {
        "language": lang.lower(),
        "language_id": language_id_map[lang.upper()],
        "tone_offset": language_tone_start_map[lang.upper()],
        "sample_rate": SAMPLING_RATE,
        "symbols": symbols,
        "phoneme_set": f"{lang.lower()}_v1",
        "n_speakers": len(spk2id),
        "spk2id": spk2id,
    }


class _OnnxWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self, x, x_lengths, sid, tone, language, bert, ja_bert,
        noise_scale, noise_scale_w, length_scale,
    ):
        o, _, _, _ = self.model.infer(
            x, x_lengths, sid, tone, language, bert, ja_bert,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
        )
        return o


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to G.pth")
    p.add_argument("--out", default="models/dittli.onnx", help="Output .onnx path")
    p.add_argument("--lang", default="EN", help="Language code (EN, DE, …)")
    p.add_argument("--spk2id", default=None,
                   help='JSON dict, e.g. \'{"MALE":0}\' — defaults match the language')
    p.add_argument("--no-fp16", action="store_true", help="Skip FP16 conversion step")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)

    if args.spk2id:
        spk2id = json.loads(args.spk2id)
    elif args.lang.upper() == "EN":
        spk2id = {"MALE": 0}
    elif args.lang.upper() == "DE":
        spk2id = {"THORSTEN": 0}
    else:
        spk2id = {}

    print(f"Loading checkpoint {args.checkpoint} ...")
    model = load_engine(args.checkpoint, device=args.device)
    model.eval()
    wrapper = _OnnxWrapper(model)
    wrapper.eval()

    seq_len = 10
    x = torch.randint(0, 100, (1, seq_len), dtype=torch.long)
    x_lengths = torch.LongTensor([seq_len])
    sid = torch.LongTensor([next(iter(spk2id.values())) if spk2id else 0])
    tone = torch.randint(0, 5, (1, seq_len), dtype=torch.long)
    language = torch.full((1, seq_len), language_id_map[args.lang.upper()], dtype=torch.long)
    bert = torch.zeros((1, 1024, seq_len), dtype=torch.float)
    ja_bert = torch.zeros((1, 768, seq_len), dtype=torch.float)
    noise_scale = torch.FloatTensor([0.667])
    noise_scale_w = torch.FloatTensor([0.8])
    length_scale = torch.FloatTensor([1.0])

    with torch.no_grad():
        out = wrapper(
            x, x_lengths, sid, tone, language, bert, ja_bert,
            noise_scale, noise_scale_w, length_scale,
        )
    print(f"sanity-check output shape: {tuple(out.shape)}")

    print(f"Exporting ONNX to {args.out} ...")
    torch.onnx.export(
        wrapper,
        (x, x_lengths, sid, tone, language, bert, ja_bert,
         noise_scale, noise_scale_w, length_scale),
        args.out,
        export_params=True,
        opset_version=14,
        dynamo=False,
        do_constant_folding=True,
        input_names=[
            "x", "x_lengths", "sid", "tone", "language", "bert", "ja_bert",
            "noise_scale", "noise_scale_w", "length_scale",
        ],
        output_names=["audio"],
        dynamic_axes={
            "x":        {0: "batch_size", 1: "text_length"},
            "tone":     {0: "batch_size", 1: "text_length"},
            "language": {0: "batch_size", 1: "text_length"},
            "bert":     {0: "batch_size", 2: "text_length"},
            "ja_bert":  {0: "batch_size", 2: "text_length"},
            "audio":    {0: "batch_size", 2: "audio_length"},
        },
    )
    fp32_size = os.path.getsize(args.out) / (1024 * 1024)
    print(f"FP32 size: {fp32_size:.2f} MB")

    sidecar = args.out.replace(".onnx", ".json")
    if not sidecar.endswith(".json"):
        sidecar = args.out + ".json"
    symbols = _resolve_symbols(args.lang)
    meta = _build_metadata(args.lang, spk2id, symbols)
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Wrote sidecar {sidecar} (n_symbols={len(symbols)}, "
          f"language_id={meta['language_id']}, tone_offset={meta['tone_offset']})")

    if args.no_fp16:
        return

    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        from onnxruntime.transformers.float16 import convert_float_to_float16

        from onnx import load_model, save_model

        print("Converting to FP16 ...")
        pre_path = args.out.replace(".onnx", "_infer.onnx")
        fp16_path = args.out.replace(".onnx", "_fp16.onnx")
        quant_pre_process(args.out, pre_path, skip_symbolic_shape=True)
        m = load_model(pre_path)
        m16 = convert_float_to_float16(m)
        save_model(m16, fp16_path)
        print(f"FP16 size: {os.path.getsize(fp16_path) / (1024 * 1024):.2f} MB")
    except ImportError as e:
        print(f"FP16 conversion skipped ({e}).")


if __name__ == "__main__":
    main()
