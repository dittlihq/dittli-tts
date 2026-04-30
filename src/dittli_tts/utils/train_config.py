"""Hyperparameters used during training (not at inference time).

Inference-time hyperparameters live in `dittli_tts.utils.config`. We keep them
separated so importing inference utilities never pulls training defaults.
"""

# Optimizer
LEARNING_RATE = 2e-4
BETAS = (0.8, 0.99)
EPS = 1e-9
LR_DECAY = 0.999875
GRAD_CLIP = 5.0

# Schedule
TOTAL_STEPS = 100_000
WARMUP_STEPS = 0
LOG_INTERVAL = 50
SAVE_INTERVAL = 1000

# Batch / segment
BATCH_SIZE = 16
SEGMENT_SIZE = 32  # in spec frames; matches SEGMENT_FRAMES in inference config

# Loss weights
C_MEL = 45.0
C_KL = 1.0
C_DUR = 1.0
C_FM = 2.0      # already baked into feature_matching_loss; here for visibility

# Mel config (must match the mel used to compute the dataset spec via FFT params)
N_MELS = 128
F_MIN = 0.0
F_MAX = None     # → sr/2

# Dataloader
NUM_WORKERS = 4

# Speakers — Thorsten Voice ships as a single speaker.
N_SPEAKERS_DE = 1
SPK2ID_DE = {"THORSTEN": 0}
