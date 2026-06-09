"""
Benchmark: PyTorch CPU  vs  ONNX Runtime CPU
Runs N_WARMUP warm-up then N_RUNS timed iterations.
Prints RTF (Real-Time Factor) comparison table.
"""

import argparse
import os
import tempfile
import time

import numpy as np
import soundfile as sf

TEXT = "The weather is nice today, and I feel very relaxed."
N_WARMUP = 2
N_RUNS = 5
_TMP_WAV = os.path.join(tempfile.gettempdir(), "_dittli_bench.wav")


def bench_pytorch(ckpt: str, device: str = "cpu"):
    from dittli_tts.inference.engine import load_engine, synthesize
    from dittli_tts.utils.config import SAMPLING_RATE

    model = load_engine(ckpt, device=device)
    model.eval()

    # warm-up
    for _ in range(N_WARMUP):
        synthesize(TEXT, _TMP_WAV, model, speaker="female", device=device)

    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        synthesize(TEXT, _TMP_WAV, model, speaker="female", device=device)
        times.append(time.perf_counter() - t0)

    return times, SAMPLING_RATE


def bench_onnx(onnx_dir: str, use_gpu: bool = False):
    from dittli_tts.inference.onnx import OnnxDittliTTS
    from dittli_tts.utils.config import SAMPLING_RATE

    engine = OnnxDittliTTS(onnx_dir=onnx_dir, use_gpu=use_gpu)

    # warm-up
    for _ in range(N_WARMUP):
        engine.speak(TEXT, output_path=_TMP_WAV)

    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        audio = engine.speak(TEXT, output_path=_TMP_WAV)
        times.append(time.perf_counter() - t0)
        audio_len = len(audio)

    sr = SAMPLING_RATE
    return times, sr, audio_len


def print_table(label, times_list, audio_secs):
    avg = np.mean(times_list)
    mn = np.min(times_list)
    mx = np.max(times_list)
    rtf = avg / audio_secs
    speed = audio_secs / avg
    print(f"  {label:<22} | avg {avg:.3f}s | min {mn:.3f}s | max {mx:.3f}s | RTF {rtf:.3f}x  (~{speed:.1f}x RT)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs ONNX inference")
    parser.add_argument("--checkpoint", "-c", default="checkpoints/G.pth")
    parser.add_argument("--onnx-dir", "-o", default="onnx")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--gpu-onnx", action="store_true", help="Use ONNX CUDAExecutionProvider")
    args = parser.parse_args()

    print(f"\n{'=' * 65}")
    print("  DittliTTS Inference Benchmark")
    print(f"  Text : {TEXT}")
    print(f"  Runs : {N_WARMUP} warm-up + {N_RUNS} timed")
    print(f"{'=' * 65}\n")

    # ── PyTorch ─────────────────────────────────────────────────────────
    print(f"[PyTorch  {args.device.upper()}]")
    pt_times, sr = bench_pytorch(args.checkpoint, device=args.device)

    # Measure audio length from a real run
    from dittli_tts.inference.engine import load_engine, synthesize

    model = load_engine(args.checkpoint, device=args.device)
    synthesize(TEXT, _TMP_WAV, model, speaker="female", device=args.device)
    audio_data, _ = sf.read(_TMP_WAV)
    audio_secs = len(audio_data) / sr
    print(f"  Audio duration: {audio_secs:.3f}s at {sr}Hz")

    # ── ONNX ─────────────────────────────────────────────────────────────
    print(f"\n[ONNX Runtime  (gpu={args.gpu_onnx})]")
    ort_times, sr2, n_samples = bench_onnx(args.onnx_dir, use_gpu=args.gpu_onnx)
    onnx_audio_secs = n_samples / sr2

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'-' * 65}")
    print(f"  {'Backend':<22} | {'avg':>7} | {'min':>7} | {'max':>7} | RTF")
    print(f"{'-' * 65}")
    print_table(f"PyTorch {args.device.upper()}", pt_times, audio_secs)
    print_table("ONNX CPU", ort_times, onnx_audio_secs)

    speedup = np.mean(pt_times) / np.mean(ort_times)
    print(f"{'-' * 65}")
    print(f"  ONNX is  {speedup:.2f}x  {'faster' if speedup > 1 else 'slower'} than PyTorch on {args.device.upper()}\n")


if __name__ == "__main__":
    main()
