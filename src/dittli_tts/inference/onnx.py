"""ONNX Runtime inference engine for DittliTTS (single-file export)."""
import os

import numpy as np
import soundfile as sf

from dittli_tts.nn import commons
from dittli_tts.text import phonemes_to_ids
from dittli_tts.text.english import grapheme_to_phoneme, normalize_text
from dittli_tts.utils.config import (
    ADD_BLANK,
    SAMPLING_RATE,
    SPK2ID,
)

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("onnxruntime is required. Run: pip install onnxruntime")


def _build_session(path: str, use_gpu: bool = False):
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_gpu else
        ["CPUExecutionProvider"]
    )
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = os.cpu_count() or 4
    return ort.InferenceSession(path, sess_options=opts, providers=providers)


class OnnxDittliTTS:
    """Inference using a single exported ONNX model file.

    Args:
        onnx_path: path to the .onnx file produced by dittli_tts.inference.export
        use_gpu:   if True, try CUDAExecutionProvider
    """

    def __init__(self, onnx_path: str = "models/dittli.onnx", use_gpu: bool = False):
        onnx_path = os.path.abspath(onnx_path)
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        print(f"Loading ONNX model: {onnx_path}")
        self._session = _build_session(onnx_path, use_gpu)
        print("ONNX session ready.")

    def _text_to_ids(self, text: str):
        normalized = normalize_text(text)
        phones, tones, _ = grapheme_to_phoneme(normalized)
        phone_ids, tone_ids, lang_ids = phonemes_to_ids(phones, tones, "EN")

        if ADD_BLANK:
            phone_ids = commons.insert_blanks(phone_ids, 0)
            tone_ids  = commons.insert_blanks(tone_ids, 0)
            lang_ids  = commons.insert_blanks(lang_ids, 0)

        return phone_ids, tone_ids, lang_ids

    def speak(
        self,
        text: str,
        output_path: str = "onnx_output.wav",
        speaker: str = "female",
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8,
        length_scale: float = 1.0,
        output_sr: int | None = None,
    ) -> np.ndarray:
        """Synthesize speech and save to output_path."""
        print(f"[ONNX] Synthesizing: {text}")

        phone_ids, tone_ids, lang_ids = self._text_to_ids(text)
        T = len(phone_ids)
        sid_val = SPK2ID.get(speaker, 0)

        feeds = {
            "x":             np.array(phone_ids, dtype=np.int64)[None, :],
            "x_lengths":     np.array([T], dtype=np.int64),
            "sid":           np.array([sid_val], dtype=np.int64),
            "tone":          np.array(tone_ids, dtype=np.int64)[None, :],
            "language":      np.array(lang_ids, dtype=np.int64)[None, :],
            "bert":          np.zeros((1, 1024, T), dtype=np.float32),
            "ja_bert":       np.zeros((1, 768, T), dtype=np.float32),
            "noise_scale":   np.array([noise_scale], dtype=np.float32),
            "noise_scale_w": np.array([noise_scale_w], dtype=np.float32),
            "length_scale":  np.array([length_scale], dtype=np.float32),
        }

        results = self._session.run(None, feeds)
        audio_np = results[0][0, 0]  # [1, 1, samples] → [samples]

        save_sr = SAMPLING_RATE
        if output_sr is not None and output_sr != SAMPLING_RATE:
            try:
                import torch
                import torchaudio
                wav_t = torch.from_numpy(audio_np).unsqueeze(0)
                audio_np = torchaudio.transforms.Resample(SAMPLING_RATE, output_sr)(wav_t).squeeze(0).numpy()
                save_sr = output_sr
            except Exception as e:
                print(f"[ONNX] Resampling failed ({e}), saving at {SAMPLING_RATE}Hz")

        sf.write(output_path, audio_np, save_sr)
        print(f"[ONNX] Saved: {output_path} ({save_sr}Hz)")
        return audio_np
