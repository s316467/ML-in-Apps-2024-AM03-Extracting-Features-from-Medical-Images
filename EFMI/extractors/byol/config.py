# config.py

BATCH_SIZE = 32
EPOCHS     = 10
LR         = 3e-5
NUM_GPUS   = 2
IMAGE_SIZE = 256
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
CHECKPOINT_EVERY = 100
CHECKPOINT_FOLDER = './checkpoints'
RESUME_FROM_CHECKPOINT = True
NUM_WORKERS = 8  # Per esempio, o usa multiprocessing.cpu_count()
PROJECTION_SIZE = 128
PROJECTION_HIDDEN_SIZE = 4096


