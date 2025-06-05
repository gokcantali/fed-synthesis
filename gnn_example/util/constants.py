from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
TRAIN_SIZE = 1 - VALIDATION_SIZE - TEST_SIZE

PROJECT_ROOT = Path(__file__).parent.parent
