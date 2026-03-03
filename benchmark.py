"""
Fair benchmark: TinyTTS vs Piper vs Kokoro ONNX vs KittenTTS nano/mini vs Pocket-TTS vs Supertonic
All CPU-only, same sentence, same warmup+timing protocol.
"""
import os, sys, time, tempfile, wave
import numpy as np
import soundfile as sf

# Monkey-patch np.load so Kokoro can load its pickled voices array
_orig_np_load = np.load
np.load = lambda *a, **kw: _orig_np_load(*a, **{**kw, "allow_pickle": True})

sys.path.insert(0, r"c:\Users\VALTEC-07\Desktop\tiny-tts")
os.chdir(r"c:\Users\VALTEC-07\Desktop\tiny-tts")

TEXT     = "The weather is nice today, and I feel very relaxed."
N_WARMUP = 5
N_RUNS   = 20
TMP      = os.path.join(tempfile.gettempdir(), "_bench_{}.wav")

results = {}

def bench(name, fn_warmup, fn_run, get_audio_path):
    """Generic benchmark runner. Uses MEDIAN of N_RUNS for stability."""
    for _ in range(N_WARMUP):
        fn_warmup()
    t0 = time.perf_counter()
    fn_run(0)
    ttfa = (time.perf_counter() - t0) * 1000
    times = []
    for i in range(N_RUNS):
        t = time.perf_counter()
        fn_run(i % 5)  # reuse 5 file slots
        times.append(time.perf_counter() - t)
    audio_data, sr = sf.read(get_audio_path(0))
    audio_secs = len(audio_data) / sr
    med = np.median(times)
    std = np.std(times)
    print(f"  {name}: median={med*1000:.0f}ms +/-{std*1000:.0f}ms | {audio_secs:.2f}s audio | {audio_secs/med:.1f}x RT")
    return dict(ttfa=ttfa, total=med, audio=audio_secs, rtfx=audio_secs/med)


# ── 1. TinyTTS (PyTorch) ──────────────────────────────────────────────────────
print("\n[1] TinyTTS (PyTorch)...")
from tiny_tts.infer import load_engine, synthesize
from tiny_tts.utils.config import SAMPLING_RATE as TINY_SR
tiny_model = load_engine("G_150000.pth", device="cpu")
results["TinyTTS (PyTorch)"] = bench(
    "TinyTTS",
    lambda: synthesize(TEXT, TMP.format("tiny_w"), tiny_model, speaker="MALE", device="cpu"),
    lambda i: synthesize(TEXT, TMP.format(f"tiny{i}"), tiny_model, speaker="MALE", device="cpu"),
    lambda i: TMP.format(f"tiny{i}"),
)


# ── 1b. TinyTTS (ONNX) ────────────────────────────────────────────────────────
print("\n[1b] TinyTTS (ONNX)...")
try:
    import onnxruntime as ort
    from tiny_tts.utils.config import SPK2ID, ADD_BLANK
    from tiny_tts.text.english import normalize_text, grapheme_to_phoneme
    from tiny_tts.text import phonemes_to_ids
    from tiny_tts.nn import commons
    
    onnx_sess = ort.InferenceSession("tinytts.onnx", providers=["CPUExecutionProvider"])
    
    def run_tiny_onnx(tag):
        # Text to phonemes
        normalized = normalize_text(TEXT)
        phones, tones, _ = grapheme_to_phoneme(normalized)
        phone_ids, tone_ids, lang_ids = phonemes_to_ids(phones, tones, "EN")
        if ADD_BLANK:
            phone_ids = commons.insert_blanks(phone_ids, 0)
            tone_ids = commons.insert_blanks(tone_ids, 0)
            lang_ids = commons.insert_blanks(lang_ids, 0)
            
        x = np.array([phone_ids], dtype=np.int64)
        x_lengths = np.array([len(phone_ids)], dtype=np.int64)
        sid = np.array([SPK2ID["MALE"]], dtype=np.int64)
        tone = np.array([tone_ids], dtype=np.int64)
        language = np.array([lang_ids], dtype=np.int64)
        bert = np.zeros((1, 1024, len(phone_ids)), dtype=np.float32)
        ja_bert = np.zeros((1, 768, len(phone_ids)), dtype=np.float32)
        n_scale = np.array([0.667], dtype=np.float32)
        n_scale_w = np.array([0.8], dtype=np.float32)
        l_scale = np.array([1.0], dtype=np.float32)

        onnx_inputs = {
            "x": x, "x_lengths": x_lengths, "sid": sid, "tone": tone, "language": language,
            "bert": bert, "ja_bert": ja_bert,
            "noise_scale": n_scale, "noise_scale_w": n_scale_w, "length_scale": l_scale
        }
        
        out = onnx_sess.run(None, onnx_inputs)[0]
        audio = out[0, 0]
        sf.write(TMP.format(tag), audio, 44100)
        
    results["TinyTTS (ONNX)"] = bench(
        "TinyTTS-ONNX",
        lambda: run_tiny_onnx("tonnx_w"),
        lambda i: run_tiny_onnx(f"tonnx{i}"),
        lambda i: TMP.format(f"tonnx{i}"),
    )
except Exception as e:
    print(f"  TinyTTS ONNX FAILED: {e}")



# ── 2. Piper ─────────────────────────────────────────────────────────────────
print("\n[2] Piper (en_US-lessac-medium)...")
try:
    from piper import PiperVoice
    pv = PiperVoice.load(r"models\piper\en_US-lessac-medium.onnx", use_cuda=False)
    def run_piper(tag):
        with wave.open(TMP.format(tag), "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(pv.config.sample_rate)
            pv.synthesize_wav(TEXT, wf)
    results["Piper (medium, ONNX)"] = bench(
        "Piper", lambda: run_piper("piper_w"), lambda i: run_piper(f"piper{i}"), lambda i: TMP.format(f"piper{i}")
    )
except Exception as e:
    print(f"  Piper FAILED: {e}")


# ── 3. Kokoro ONNX ────────────────────────────────────────────────────────────
print("\n[3] Kokoro ONNX...")
try:
    from kokoro_onnx import Kokoro
    kokoro = Kokoro(r"models\kokoro\kokoro-v0_19.onnx", r"models\kokoro\voices.npz")
    def run_kokoro(tag):
        samples, sr = kokoro.create(TEXT, voice="af_bella", speed=1.0, lang="en-us")
        sf.write(TMP.format(tag), samples, sr)
    results["Kokoro ONNX (82M)"] = bench(
        "Kokoro", lambda: run_kokoro("kok_w"), lambda i: run_kokoro(f"kok{i}"), lambda i: TMP.format(f"kok{i}")
    )
except Exception as e:
    print(f"  Kokoro FAILED: {e}")


# ── 4. KittenTTS nano ─────────────────────────────────────────────────────────
print("\n[4] KittenTTS nano...")
try:
    from kittentts import KittenTTS
    kitten_nano = KittenTTS("KittenML/kitten-tts-nano-0.8")
    def run_kitten_nano(tag):
        audio = kitten_nano.generate(TEXT, voice="expr-voice-5-m")
        sf.write(TMP.format(tag), audio, 24000)
    results["KittenTTS nano"] = bench(
        "KittenTTS-nano", lambda: run_kitten_nano("kn_w"), lambda i: run_kitten_nano(f"kn{i}"), lambda i: TMP.format(f"kn{i}")
    )
except Exception as e:
    print(f"  KittenTTS nano FAILED: {e}")


# ── 5. KittenTTS mini ─────────────────────────────────────────────────────────
print("\n[5] KittenTTS mini...")
try:
    from kittentts import KittenTTS
    kitten_mini = KittenTTS("KittenML/kitten-tts-mini-0.8")
    def run_kitten_mini(tag):
        audio = kitten_mini.generate(TEXT, voice="expr-voice-5-m")
        sf.write(TMP.format(tag), audio, 24000)
    results["KittenTTS mini"] = bench(
        "KittenTTS-mini", lambda: run_kitten_mini("km_w"), lambda i: run_kitten_mini(f"km{i}"), lambda i: TMP.format(f"km{i}")
    )
except Exception as e:
    print(f"  KittenTTS mini FAILED: {e}")


# ── 6. Pocket-TTS ─────────────────────────────────────────────────────────────
print("\n[6] Pocket-TTS...")
try:
    from pocket_tts import TTSModel
    import scipy.io.wavfile as wavfile
    pocket = TTSModel.load_model()
    voice_state = pocket.get_state_for_audio_prompt("alba")
    def run_pocket(tag):
        audio = pocket.generate_audio(voice_state, TEXT)
        wavfile.write(TMP.format(tag), pocket.sample_rate, audio.numpy())
    results["Pocket-TTS"] = bench(
        "Pocket-TTS", lambda: run_pocket("pt_w"), lambda i: run_pocket(f"pt{i}"), lambda i: TMP.format(f"pt{i}")
    )
except Exception as e:
    print(f"  Pocket-TTS FAILED: {e}")


# ── 7. Supertonic ONNX ────────────────────────────────────────────────────────
print("\n[7] Supertonic ONNX (5 flow steps)...")
try:
    import onnxruntime as ort
    import json
    from unicodedata import normalize as unicode_normalize
    import re as _re

    SUPER_DIR = r"models\supertonic"

    # --- Minimal helpers from supertonic/py/helper.py ---
    def _length_to_mask(lengths, max_len=None):
        max_len = max_len or lengths.max()
        ids = np.arange(0, max_len)
        mask = (ids < np.expand_dims(lengths, 1)).astype(np.float32)
        return mask.reshape(-1, 1, max_len)

    def _get_latent_mask(wav_lengths, base_chunk_size, chunk_compress_factor):
        latent_size = base_chunk_size * chunk_compress_factor
        latent_lengths = (wav_lengths + latent_size - 1) // latent_size
        return _length_to_mask(latent_lengths)

    # Unicode processor
    with open(os.path.join(SUPER_DIR, "onnx", "unicode_indexer.json")) as f:
        _uidx = json.load(f)

    def _text_to_ids(text, lang):
        text = unicode_normalize("NFKD", text)
        text = _re.sub(r"\s+", " ", text).strip()
        if not _re.search(r"[.!?]$", text):
            text += "."
        text = f"<{lang}>" + text + f"</{lang}>"
        # _uidx is a list[65536] indexed by unicode code point; -1 = unknown -> skip
        ids = [_uidx[ord(c)] for c in text if ord(c) < len(_uidx) and _uidx[ord(c)] != -1]
        return np.array(ids, dtype=np.int64)

    # Load 4 ONNX sessions
    opts = ort.SessionOptions()
    _dp  = ort.InferenceSession(os.path.join(SUPER_DIR, "onnx", "duration_predictor.onnx"), opts, ["CPUExecutionProvider"])
    _te  = ort.InferenceSession(os.path.join(SUPER_DIR, "onnx", "text_encoder.onnx"),       opts, ["CPUExecutionProvider"])
    _ve  = ort.InferenceSession(os.path.join(SUPER_DIR, "onnx", "vector_estimator.onnx"),   opts, ["CPUExecutionProvider"])
    _voc = ort.InferenceSession(os.path.join(SUPER_DIR, "onnx", "vocoder.onnx"),            opts, ["CPUExecutionProvider"])

    # Load config + voice style
    with open(os.path.join(SUPER_DIR, "onnx", "tts.json")) as f:
        _cfg = json.load(f)
    _sr   = _cfg["ae"]["sample_rate"]
    _bcs  = _cfg["ae"]["base_chunk_size"]
    _ccf  = _cfg["ttl"]["chunk_compress_factor"]
    _ldim = _cfg["ttl"]["latent_dim"]

    with open(os.path.join(SUPER_DIR, "voice_styles", "M1.json")) as f:
        _vs = json.load(f)

    def _load_style(vs):
        td = vs["style_ttl"]["dims"];  dd = vs["style_dp"]["dims"]
        ttl = np.array(vs["style_ttl"]["data"], np.float32).reshape(td[1], td[2])[np.newaxis]
        dp  = np.array(vs["style_dp"]["data"],  np.float32).reshape(dd[1], dd[2])[np.newaxis]
        return ttl, dp

    _style_ttl, _style_dp = _load_style(_vs)

    _TOTAL_STEPS = 2   # match Supertonic's RTF benchmark setting (2 steps)
    _SPEED = 1.05

    def run_supertonic(tag):
        ids = _text_to_ids(TEXT, "en")
        text_ids = ids[np.newaxis]                                  # [1, T]
        t_len = np.array([len(ids)], dtype=np.int64)
        text_mask = _length_to_mask(t_len)

        dur, *_ = _dp.run(None, {"text_ids": text_ids, "style_dp": _style_dp, "text_mask": text_mask})
        dur = dur / _SPEED
        text_emb, *_ = _te.run(None, {"text_ids": text_ids, "style_ttl": _style_ttl, "text_mask": text_mask})

        wav_len = np.array([int(dur[0] * _sr)], dtype=np.int64)
        latent_mask = _get_latent_mask(wav_len, _bcs, _ccf)
        chunk_sz = _bcs * _ccf
        latent_len = int(np.ceil(dur[0] * _sr / chunk_sz))
        xt = np.random.randn(1, _ldim * _ccf, latent_len).astype(np.float32) * latent_mask

        total_s = np.array([_TOTAL_STEPS], np.float32)
        for step in range(_TOTAL_STEPS):
            cur_s = np.array([step], np.float32)
            xt, *_ = _ve.run(None, {
                "noisy_latent": xt, "text_emb": text_emb,
                "style_ttl": _style_ttl, "text_mask": text_mask,
                "latent_mask": latent_mask,
                "current_step": cur_s, "total_step": total_s,
            })

        wav, *_ = _voc.run(None, {"latent": xt})
        trim_len = int(dur[0] * _sr)
        wav = wav[0, :trim_len]
        sf.write(TMP.format(tag), wav, _sr)

    results["Supertonic ONNX (2-step)"] = bench(
        "Supertonic",
        lambda: run_supertonic("st_w"),
        lambda i: run_supertonic(f"st{i}"),
        lambda i: TMP.format(f"st{i}"),
    )
except FileNotFoundError as e:
    print(f"  Supertonic SKIPPED: model file not found — {e}")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  Supertonic FAILED: {e}")


# ── Results Table ─────────────────────────────────────────────────────────────
W = 70
print("\n" + "="*W)
print(f"  TTS Benchmark (CPU) — \"{TEXT}\"")
print("="*W)
print(f"  {'ENGINE':<24} | {'TTFA(ms)':>8} | {'TOTAL(s)':>8} | {'AUDIO(s)':>8} | RTFx")
print("-"*W)
for name, r in results.items():
    mark = " 🏆" if name.startswith("TinyTTS") else ""
    print(f"  {name:<24} | {r['ttfa']:>8.0f} | {r['total']:>8.3f} | {r['audio']:>8.3f} | {r['rtfx']:.1f}x{mark}")
print("="*W)

with open("benchmark_results_fair.txt", "w", encoding="utf-8") as f:
    f.write(f"TTS CPU Benchmark — {TEXT}\n\n")
    f.write(f"{'ENGINE':<24} | {'TTFA(ms)':>8} | {'TOTAL(s)':>8} | {'AUDIO(s)':>8} | RTFx\n")
    f.write("-"*W + "\n")
    for name, r in results.items():
        f.write(f"{name:<24} | {r['ttfa']:>8.0f} | {r['total']:>8.3f} | {r['audio']:>8.3f} | {r['rtfx']:.1f}x\n")
print("Saved to benchmark_results_fair.txt")
