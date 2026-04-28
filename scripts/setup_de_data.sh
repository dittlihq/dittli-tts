#!/usr/bin/env bash
# Download Thorsten Voice (~3.7 GB CC0 German TTS dataset) and the English
# warm-start checkpoint. Idempotent — safe to re-run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data/thorsten"
CKPT_DIR="$REPO_ROOT/checkpoints"

mkdir -p "$DATA_DIR" "$CKPT_DIR"

# ---- 1. Thorsten Voice dataset (CC0) -----------------------------------
# Mirror: https://www.openslr.org/resources/95/ ; HF: Thorsten-Voice/TTS-Dataset
# Using the 22.05 kHz "Neutral" subset (~23 h). Resampled to 44.1 kHz at
# preprocess time; the audio module handles that automatically.
THORSTEN_TGZ="$DATA_DIR/ThorstenVoice-Dataset_2022.10.tgz"
THORSTEN_URL="https://www.openslr.org/resources/95/ThorstenVoice-Dataset_2022.10.tgz"

if [ ! -d "$DATA_DIR/wavs" ] || [ ! -f "$DATA_DIR/metadata.csv" ]; then
    if [ ! -f "$THORSTEN_TGZ" ]; then
        echo "[setup] downloading Thorsten Voice (~3.7 GB) ..."
        curl -L --fail --retry 4 --retry-delay 5 -o "$THORSTEN_TGZ" "$THORSTEN_URL"
    fi
    echo "[setup] extracting Thorsten Voice ..."
    tar -xzf "$THORSTEN_TGZ" -C "$DATA_DIR" --strip-components=1
    # Some Thorsten releases ship as `ThorstenVoice-Dataset_2022.10/{wavs,metadata.csv}`.
    # The --strip-components=1 above flattens them into $DATA_DIR.
    rm -f "$THORSTEN_TGZ"
else
    echo "[setup] Thorsten dataset already present at $DATA_DIR"
fi

# Sanity check
if [ ! -f "$DATA_DIR/metadata.csv" ]; then
    echo "[setup] ERROR: metadata.csv not found after extract. Inspect $DATA_DIR." >&2
    exit 1
fi
N_WAVS=$(find "$DATA_DIR/wavs" -name "*.wav" | wc -l)
echo "[setup] dataset OK — $N_WAVS wav files, $(wc -l < "$DATA_DIR/metadata.csv") rows in metadata.csv"

# ---- 2. English warm-start checkpoint ----------------------------------
EN_CKPT="$CKPT_DIR/G.pth"
if [ ! -f "$EN_CKPT" ]; then
    echo "[setup] downloading English G.pth from HuggingFace (~6 MB) ..."
    python -c "
from huggingface_hub import hf_hub_download
import shutil, os
p = hf_hub_download(repo_id='backtracking/tiny-tts', filename='G.pth')
os.makedirs('$CKPT_DIR', exist_ok=True)
shutil.copy(p, '$EN_CKPT')
print('[setup] saved', '$EN_CKPT')
"
else
    echo "[setup] English checkpoint already present at $EN_CKPT"
fi

# ---- 3. Symbol snapshot (used by the embedding remapper) ---------------
SYM_SNAP="$CKPT_DIR/symbols_v1_en.txt"
if [ ! -f "$SYM_SNAP" ]; then
    echo "[setup] WARN: $SYM_SNAP missing — without it, the German fine-tune"
    echo "             will skip the phoneme embedding row instead of remapping."
else
    echo "[setup] symbol snapshot present ($(wc -l < "$SYM_SNAP") symbols)"
fi

echo "[setup] done. Next: pre-compute features"
echo "             python -m tiny_tts.data.preprocess \\"
echo "                 --metadata $DATA_DIR/metadata.csv \\"
echo "                 --wavs-dir $DATA_DIR/wavs"
