"""Modal entrypoint for German fine-tuning on Thorsten Voice.

One-time setup (locally):
    pip install modal
    modal token new

Run training (from the repo root):
    modal run --detach src/dittli_tts/training/modal.py                  # full run on A10G
    modal run src/dittli_tts/training/modal.py --max-steps 200           # quick smoke

Reconnect / inspect a detached run:
    modal app logs dittli-de-train
    modal volume ls dittli-de
    modal volume get dittli-de checkpoints_de/G_<step>.pth ./G_de.pth

Resume after a previous run was killed (timeout, OOM, manual stop):
    Just re-run the same `modal run` command. The volume keeps prior
    checkpoints; this script auto-detects the highest-step `G_*.pth` and
    warm-starts from it. The English warm-start is only used on the very
    first run.

Cost on A10G (~$1.10/hr): roughly $5–10 for a usable model (~50 k steps),
$10–20 for full polish (~100 k steps). Fits in a $30/month free credit.
Override the GPU at the call site by editing GPU_KIND below.
"""

from __future__ import annotations

import modal

GPU_KIND = "A10G"  # cheaper: "T4". Faster: "A100-40GB" or "L4".
TIMEOUT_HOURS = 12
APP_NAME = "dittli-de-train"
VOLUME_NAME = "dittli-de"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "ffmpeg", "unzip")
    .uv_sync()
    .add_local_dir(
        ".",
        remote_path="/root/dittli-tts",
        ignore=[
            "data/**",
            "checkpoints/*.pth",
            "checkpoints_de/**",
            "venv/**",
            ".venv/**",
            ".git/**",
            "__pycache__/**",
            "node_modules/**",
            "models/*.onnx",
            "*.pth",
            "*.zip",
            "*.tgz",
            "*.tar.gz",
        ],
    )
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)


@app.function(
    gpu=GPU_KIND,
    timeout=TIMEOUT_HOURS * 60 * 60,
    volumes={"/root/ckpt-vol": volume},
)
def train(max_steps: int | None = None, batch_size: int = 8) -> None:
    import glob
    import os
    import shutil
    import subprocess
    import sys

    os.chdir("/root/dittli-tts")

    # Big-but-ephemeral data goes to /tmp; only checkpoints sit on the
    # persistent volume. We symlink so the existing scripts don't need
    # any new flags.
    os.makedirs("/tmp/thorsten", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    if os.path.lexists("data/thorsten"):
        if os.path.islink("data/thorsten"):
            os.remove("data/thorsten")
        else:
            shutil.rmtree("data/thorsten")
    os.symlink("/tmp/thorsten", "data/thorsten")

    for d in ("checkpoints", "checkpoints_de"):
        target = f"/root/ckpt-vol/{d}"
        os.makedirs(target, exist_ok=True)
        if os.path.lexists(d):
            if os.path.islink(d):
                os.remove(d)
            else:
                for f in os.listdir(d):
                    src = os.path.join(d, f)
                    dst = os.path.join(target, f)
                    if not os.path.exists(dst):
                        shutil.move(src, dst)
                shutil.rmtree(d, ignore_errors=True)
        os.symlink(target, d)

    print("[modal] running setup_de_data.sh ...")
    subprocess.run(["bash", "scripts/setup_de_data.sh"], check=True)

    print("[modal] running preprocess (~10 min) ...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "dittli_tts.data.preprocess",
            "--metadata",
            "data/thorsten/metadata.csv",
            "--wavs-dir",
            "data/thorsten/wavs",
        ],
        check=True,
    )

    # Resume-from-volume: pick the highest-step German checkpoint if any,
    # else fall back to the English warm-start. Non-numeric step names
    # (e.g. G_final.pth) are skipped — the trainer always saves a numbered
    # checkpoint at the same step as G_final.pth, so we lose nothing.
    def _ckpt_step(path: str) -> int:
        suffix = os.path.basename(path)[len("G_") : -len(".pth")]
        return int(suffix) if suffix.isdigit() else -1

    de_ckpts = sorted(
        (p for p in glob.glob("checkpoints_de/G_*.pth") if _ckpt_step(p) >= 0),
        key=_ckpt_step,
    )
    extra_args: list[str] = []
    if de_ckpts:
        warm_start = de_ckpts[-1]
        # Skip embedding remap on resume — the German checkpoint is already
        # in the new (220-symbol) layout. Pointing --old-symbols at a
        # non-existent path triggers finetune_de.py's "no remap" branch.
        extra_args += ["--old-symbols", "/nonexistent/skip-remap"]
        print(f"[modal] RESUMING from {warm_start}")
    else:
        warm_start = "checkpoints/G.pth"
        print(f"[modal] starting fresh, warm-start: {warm_start}")

    cmd = [
        sys.executable,
        "scripts/finetune_de.py",
        "--metadata",
        "data/thorsten/metadata.csv",
        "--wavs-dir",
        "data/thorsten/wavs",
        "--english-ckpt",
        warm_start,
        "--ckpt-dir",
        "checkpoints_de/",
        "--batch-size",
        str(batch_size),
        "--device",
        "cuda",
        *extra_args,
    ]
    if max_steps is not None:
        cmd += ["--max-steps", str(max_steps)]
    print(f"[modal] launching: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    finally:
        # Flush even on crash / OOM / timeout — partial progress is useful.
        volume.commit()

    final_ckpts = sorted(glob.glob("checkpoints_de/G_*.pth"))
    print(f"[modal] saved {len(final_ckpts)} checkpoints to volume:")
    for p in final_ckpts[-5:]:
        print(f"  {p}  ({os.path.getsize(p) / 1e6:.1f} MB)")


@app.local_entrypoint()
def main(max_steps: int | None = None, batch_size: int = 8) -> None:
    train.remote(max_steps=max_steps, batch_size=batch_size)
