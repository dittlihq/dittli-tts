# German TTS Fine-Tuning Guide

This walks through fine-tuning TinyTTS on German (Thorsten Voice). The full
run takes ~24 h on an A100 or ~3 days on a 3090 (~50–100 k steps). Smoke
tests before committing to the full run are highly recommended.

## TL;DR

```bash
# 1. Install training deps + audio tools
pip install -r requirements.txt
sudo apt-get install -y ffmpeg sox

# 2. Pull the Thorsten dataset (~3.7 GB) and the English warm-start checkpoint
bash scripts/setup_de_data.sh

# 3. Pre-compute spectrograms + phoneme IDs (one-off, ~10 min CPU)
python -m tiny_tts.data.preprocess \
    --metadata data/thorsten/metadata.csv \
    --wavs-dir data/thorsten/wavs

# 4. Smoke test (CPU is fine, ~30 s)
python scripts/smoke_de.py --metadata data/thorsten/metadata.csv \
    --wavs-dir data/thorsten/wavs

# 5. Real training run (GPU)
python scripts/finetune_de.py \
    --metadata data/thorsten/metadata.csv \
    --wavs-dir data/thorsten/wavs \
    --english-ckpt checkpoints/G.pth \
    --ckpt-dir checkpoints_de/ \
    --device cuda
```

---

## Where to run if you don't own a GPU

| Provider | Pricing | Best for | Notes |
|---|---|---|---|
| **Vast.ai** | $0.20–$1.00/h for RTX 3090–4090 | Cheapest A100/3090 by far | Bring-your-own image; needs a debit card |
| **RunPod** | ~$0.40/h RTX 3090, $1.50/h A100 | Easy templates, persistent volumes | Pre-built PyTorch images |
| **Lambda Labs** | $1.10/h A100 (40 GB), $1.99/h H100 | Reliable, ssh-friendly | Capacity sometimes scarce |
| **Modal** | Pay-per-second A10G $1.10/h, A100 $4/h | One-shot training jobs from Python | Easiest if you're already a Python person |
| **Google Colab Pro+** | $50/mo | Quick experiments | A100s available but not guaranteed; 24 h runtime cap |
| **Kaggle Notebooks** | Free, 30 h/week of T4 or P100 | Rapid prototyping | Cannot leave running unattended |
| **AWS / GCP / Azure** | A100 ~$3–5/h | If you're already there | Highest setup cost |

For Thorsten Voice end-to-end, **a single A100 for ~24 h (~$25–$50)** is the
most cost-effective path. A 3090 for 3 days (~$30–$70 on Vast.ai) is the
cheapest absolute number.

---

## Cloud-by-cloud quickstart

### Vast.ai (cheapest)

1. Sign up at vast.ai, add $20 credit.
2. Search → filter `GPU Total RAM ≥ 24 GB`, sort by `$/hr`.
3. Pick a `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` template, 50 GB disk.
4. SSH in, then:
   ```bash
   git clone https://github.com/brio1009/tiny-tts.git
   cd tiny-tts
   git checkout claude/german-implementation-progress-15W8v   # or wherever the German branch lives
   pip install -r requirements.txt
   bash scripts/setup_de_data.sh
   python -m tiny_tts.data.preprocess \
       --metadata data/thorsten/metadata.csv \
       --wavs-dir data/thorsten/wavs
   nohup python scripts/finetune_de.py \
       --metadata data/thorsten/metadata.csv \
       --wavs-dir data/thorsten/wavs \
       --english-ckpt checkpoints/G.pth \
       --ckpt-dir checkpoints_de/ \
       > training.log 2>&1 &
   tail -f training.log
   ```
5. After training, scp the latest `checkpoints_de/G_*.pth` back to your laptop.

### RunPod

1. Pick a *PyTorch 2.1* template (A40 / 3090 / A100), 50 GB persistent volume.
2. Open the web terminal and follow the same commands as Vast.ai.
3. RunPod auto-pauses; if the pod stops, your `checkpoints_de/` survives if
   it's on the persistent volume mount.

### Modal (Python-native)

Best if you're comfortable in Python and want one command. Save the script
below as `modal_train.py`:

```python
import modal

image = (
    modal.Image.debian_slim()
    .apt_install("git", "ffmpeg")
    .pip_install_from_requirements("requirements.txt")
    .run_commands(
        "git clone https://github.com/brio1009/tiny-tts.git /root/tiny-tts",
    )
)
volume = modal.Volume.from_name("tinytts-de", create_if_missing=True)
app = modal.App("tinytts-de-train", image=image)

@app.function(gpu="A100-40GB", timeout=24 * 60 * 60, volumes={"/root/tiny-tts/checkpoints_de": volume})
def train():
    import subprocess, os
    os.chdir("/root/tiny-tts")
    subprocess.run(["bash", "scripts/setup_de_data.sh"], check=True)
    subprocess.run([
        "python", "-m", "tiny_tts.data.preprocess",
        "--metadata", "data/thorsten/metadata.csv",
        "--wavs-dir", "data/thorsten/wavs",
    ], check=True)
    subprocess.run([
        "python", "scripts/finetune_de.py",
        "--metadata", "data/thorsten/metadata.csv",
        "--wavs-dir", "data/thorsten/wavs",
        "--english-ckpt", "checkpoints/G.pth",
        "--ckpt-dir", "checkpoints_de/",
    ], check=True)

if __name__ == "__main__":
    with app.run():
        train.remote()
```

```bash
pip install modal
modal token new
modal run modal_train.py
```

When done, the checkpoint is in the named volume. Pull it:
```bash
modal volume get tinytts-de checkpoints_de/G_final.pth ./G_de.pth
```

### Google Colab

The free tier rarely gets you an A100 — expect a T4. Training will be
roughly 4–5× slower than an A100 (so ~5 days for full convergence), but
you can still get partial results in 12 h sessions.

Open a fresh Colab notebook with GPU runtime, then:

```python
!git clone https://github.com/brio1009/tiny-tts.git
%cd tiny-tts
!git checkout claude/german-implementation-progress-15W8v
!pip install -q -r requirements.txt
!bash scripts/setup_de_data.sh
!python -m tiny_tts.data.preprocess \
    --metadata data/thorsten/metadata.csv \
    --wavs-dir data/thorsten/wavs

# Mount Drive so checkpoints survive runtime resets
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/tinytts-de
!python scripts/finetune_de.py \
    --metadata data/thorsten/metadata.csv \
    --wavs-dir data/thorsten/wavs \
    --english-ckpt checkpoints/G.pth \
    --ckpt-dir /content/drive/MyDrive/tinytts-de \
    --batch-size 8         # T4 has less VRAM than A100; halve the batch
```

If the runtime resets, just re-run: `Trainer` will warm-start from the
latest `G_*.pth` you pass via `--english-ckpt /content/drive/MyDrive/tinytts-de/G_<step>.pth`.

---

## Hyperparameters (already set — only tweak if you know why)

`tiny_tts/utils/train_config.py`:

| Setting | Default | Notes |
|---|---|---|
| `BATCH_SIZE` | 16 | Use 8 on a 16-GB card (T4, V100), 24 on A100 80 GB |
| `LEARNING_RATE` | 2e-4 | Halve for very stable runs |
| `LR_DECAY` | 0.999875 | Per-epoch exponential |
| `TOTAL_STEPS` | 100 000 | 50 k often sufficient post-fine-tune |
| `C_MEL` / `C_KL` / `C_DUR` | 45 / 1 / 1 | Standard VITS recipe |
| `SEGMENT_SIZE` | 32 frames | Matches the inference config |

The trainer saves `G_<step>.pth` and `D_<step>.pth` every 1000 steps to
`--ckpt-dir`. Keep the last 3–5; delete older ones to save disk.

---

## Verifying the German checkpoint

After training:

```bash
# 1. Synthesize directly from PyTorch
python -m tiny_tts.infer \
    --lang DE \
    --checkpoint checkpoints_de/G_final.pth \
    --text "Guten Morgen, wie geht es dir heute?" \
    --output morgen.wav --device cuda

# 2. Export to ONNX (FP32 ~6 MB, FP16 ~3 MB) + sidecar JSON
python export_onnx.py \
    --checkpoint checkpoints_de/G_final.pth \
    --lang DE \
    --out models/tinytts-de.onnx

# 3. Run via the npm-package (no Python at all)
cd npm-package
node bin/cli.js "Guten Morgen, wie geht es dir?" \
    --model ../models/tinytts-de.onnx -o de.wav
```

The third step is the canonical browser/Node.js inference path that
ships in the `tiny-tts` npm package.

---

## Troubleshooting

- **OOM during training** → drop `--batch-size`. Half is usually enough
  to fit on the next GPU class down.
- **Loss explodes (NaN)** → MAS noise too high early; lower
  `mas_noise_scale_initial` in `tiny_tts/utils/config.py:MODEL_PARAMS`
  to `0.005`.
- **`No module named 'numba'`** → `pip install numba`. Required by the
  Viterbi alignment kernel (`tiny_tts/alignment/core.py`).
- **Audio sounds robotic / muffled after 100 k steps** → the discriminator
  may be over-fitting; try halving `LR_DECAY` so it decays faster, or
  reduce `C_MEL` to `30`.
- **Phoneme parity check fails after editing `german.py`** → re-run
  `python scripts/gen_de_rules.py` to regenerate the JS rules JSON, then
  `python scripts/test_g2p_parity.py` should pass again.
- **`Permission denied` writing to `models/`** → that path is in `.gitignore`
  for `.pth`/`.onnx` files; the metadata `.json` sidecars are explicitly
  whitelisted. If you keep weights outside the repo, no problem.

---

## What success looks like

After ~50 k steps you should hear:
- Clear, intelligible German with stable pitch.
- Correct stress on common words (`gehen`, `sprechen`, `Universität`).
- Slight artifacts on rare loanwords (`Genre`, `Café`) — acceptable for v1.

After ~100 k steps, the discriminator + feature-matching loss should
drive the audio to be largely indistinguishable from Thorsten's voice
on dataset-style sentences.
