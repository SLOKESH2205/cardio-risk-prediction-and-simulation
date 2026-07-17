"""Central configuration for the cardiovascular risk platform."""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5

TARGET_COLUMN = "target"
SOURCE_COLUMN = "source"
