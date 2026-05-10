#!/usr/bin/env bash
# Download LJSpeech (~2.6 GB, public domain) and prepare it for training.
# Idempotent — safe to re-run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data/ljspeech"
CKPT_DIR="$REPO_ROOT/checkpoints"

mkdir -p "$DATA_DIR" "$CKPT_DIR"

# ---- 1. LJSpeech dataset (public domain) --------------------------------
ARCHIVE="$DATA_DIR/LJSpeech-1.1.tar.bz2"
URL="https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"

if [ ! -d "$DATA_DIR/wavs" ] || [ ! -f "$DATA_DIR/metadata.csv" ]; then
    if [ ! -f "$ARCHIVE" ]; then
        echo "[setup] downloading LJSpeech-1.1 (~2.6 GB) ..."
        curl -L --fail --retry 4 --retry-delay 5 -o "$ARCHIVE" "$URL"
    fi

    echo "[setup] extracting $(basename "$ARCHIVE") ..."
    tar -xjf "$ARCHIVE" -C "$DATA_DIR" --strip-components=1
    rm -f "$ARCHIVE"
else
    echo "[setup] LJSpeech dataset already present at $DATA_DIR"
fi

# Sanity check
if [ ! -f "$DATA_DIR/metadata.csv" ]; then
    echo "[setup] ERROR: metadata.csv not found after extract. Inspect $DATA_DIR." >&2
    exit 1
fi
N_WAVS=$(find "$DATA_DIR/wavs" -name "*.wav" | wc -l)
echo "[setup] dataset OK — $N_WAVS wav files, $(wc -l < "$DATA_DIR/metadata.csv") rows in metadata.csv"

# ---- 2. English checkpoint ----------------------------------------------
EN_CKPT="$CKPT_DIR/G.pth"
if [ ! -f "$EN_CKPT" ]; then
    echo "[setup] WARN: $EN_CKPT not found. Training will start from random init." >&2
else
    echo "[setup] English checkpoint present at $EN_CKPT"
fi

echo "[setup] done. Next: pre-compute features"
echo "             python -m dittli_tts.data.preprocess \\"
echo "                 --metadata $DATA_DIR/metadata.csv \\"
echo "                 --wavs-dir $DATA_DIR/wavs \\"
echo "                 --language EN"
echo ""
echo "        Then train:"
echo "             python scripts/train_en.py \\"
echo "                 --metadata $DATA_DIR/metadata.csv \\"
echo "                 --wavs-dir $DATA_DIR/wavs"
