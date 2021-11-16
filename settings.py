import os
from pathlib import Path

# dirs setup
PROJECT_DIR = Path(__file__).parent.resolve()
LOGS_DIR = PROJECT_DIR / 'logs'
CHECKPOINTS_DIR = PROJECT_DIR / 'checkpoints'

if not os.path.exists(PROJECT_DIR):
    os.mkdir(PROJECT_DIR)
if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)
if not os.path.exists(CHECKPOINTS_DIR):
    os.mkdir(CHECKPOINTS_DIR)
