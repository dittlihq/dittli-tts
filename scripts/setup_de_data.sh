#!/usr/bin/env bash
# Download Thorsten Voice (~3.7 GB CC0 German TTS dataset) and the English
# warm-start checkpoint. Idempotent — safe to re-run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data/thorsten"
CKPT_DIR="$REPO_ROOT/checkpoints"

mkdir -p "$DATA_DIR" "$CKPT_DIR"

# ---- 1. Thorsten Voice dataset (CC0) -----------------------------------
# Primary: Zenodo 2022.10 release (~1.4 GB zip).
# Fallback: OpenSLR mirror of the older v02 release (~3.0 GB tgz).
# Both ship 22.05 kHz wavs; preprocess resamples to 44.1 kHz.
PRIMARY_ARCHIVE="$DATA_DIR/ThorstenVoice-Dataset_2022.10.zip"
PRIMARY_URL="https://zenodo.org/records/7265581/files/ThorstenVoice-Dataset_2022.10.zip"
FALLBACK_ARCHIVE="$DATA_DIR/thorsten-de_v02.tgz"
FALLBACK_URL="https://openslr.trmal.net/resources/95/thorsten-de_v02.tgz"

flatten_dataset() {
    # If extraction left a single nested top-level dir containing wavs/,
    # move its contents up to $DATA_DIR. Also strip macOS resource-fork dirs
    # that ship inside the Zenodo zip.
    # Uses shell globbing rather than `find`, which by default refuses to
    # descend into a symlinked starting-point dir (e.g. when $DATA_DIR is
    # symlinked to ephemeral storage like /tmp/thorsten).
    rm -rf "$DATA_DIR/__MACOSX"
    if [ -d "$DATA_DIR/wavs" ]; then
        return
    fi
    local top=""
    shopt -s nullglob
    for d in "$DATA_DIR"/*/; do
        if [ -d "${d}wavs" ]; then
            top="${d%/}"
            break
        fi
    done
    shopt -u nullglob
    if [ -n "$top" ]; then
        echo "[setup] flattening $(basename "$top")/ into $DATA_DIR ..."
        shopt -s dotglob
        mv "$top"/* "$DATA_DIR"/
        shopt -u dotglob
        rmdir "$top" 2>/dev/null || true
    else
        echo "[setup] flatten_dataset: no nested wavs/ dir found under $DATA_DIR" >&2
        ls -la "$DATA_DIR" >&2 || true
    fi
}

ensure_metadata_csv() {
    # The 2022.10 release ships split metadata files. Concatenate them into
    # a single metadata.csv so the default --metadata path used by
    # preprocess / smoke_de / finetune_de just works.
    if [ -f "$DATA_DIR/metadata.csv" ]; then
        return
    fi
    local merged="$DATA_DIR/metadata.csv"
    : > "$merged"
    local found=0
    for split in metadata_train.csv metadata_dev.csv metadata_test.csv; do
        if [ -f "$DATA_DIR/$split" ]; then
            cat "$DATA_DIR/$split" >> "$merged"
            found=1
        fi
    done
    if [ "$found" = "1" ]; then
        echo "[setup] merged train/dev/test splits into metadata.csv ($(wc -l < "$merged") rows)"
    else
        rm -f "$merged"
    fi
}

# Recover already-extracted state (e.g. previous run got the layout wrong).
flatten_dataset
ensure_metadata_csv

if [ ! -d "$DATA_DIR/wavs" ] || [ ! -f "$DATA_DIR/metadata.csv" ]; then
    archive=""
    if [ -f "$PRIMARY_ARCHIVE" ]; then
        archive="$PRIMARY_ARCHIVE"
    elif [ -f "$FALLBACK_ARCHIVE" ]; then
        archive="$FALLBACK_ARCHIVE"
    else
        echo "[setup] downloading Thorsten Voice 2022.10 from Zenodo (~1.4 GB) ..."
        if curl -L --fail --retry 4 --retry-delay 5 -o "$PRIMARY_ARCHIVE" "$PRIMARY_URL"; then
            archive="$PRIMARY_ARCHIVE"
        else
            echo "[setup] Zenodo download failed; trying OpenSLR v02 fallback (~3.0 GB) ..."
            rm -f "$PRIMARY_ARCHIVE"
            curl -L --fail --retry 4 --retry-delay 5 -o "$FALLBACK_ARCHIVE" "$FALLBACK_URL"
            archive="$FALLBACK_ARCHIVE"
        fi
    fi

    echo "[setup] extracting $(basename "$archive") ..."
    case "$archive" in
        *.zip)
            command -v unzip >/dev/null 2>&1 || {
                echo "[setup] ERROR: unzip not found. Install it (apt-get install -y unzip) and re-run." >&2
                exit 1
            }
            unzip -q -o "$archive" -d "$DATA_DIR"
            ;;
        *.tgz|*.tar.gz)
            tar -xzf "$archive" -C "$DATA_DIR"
            ;;
        *)
            echo "[setup] ERROR: unknown archive format: $archive" >&2
            exit 1
            ;;
    esac
    flatten_dataset
    ensure_metadata_csv
    rm -f "$archive"
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
