"""Shared constants and utility helpers."""

import os

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

INPUT_PATH: str = "input"
OUTPUT_PATH: str = "output"
MODELS_PATH: str = "models"

PROTOTXT_PATH: str = os.path.join(MODELS_PATH, "MobileNetSSD_deploy.prototxt")
MODEL_PATH: str = os.path.join(MODELS_PATH, "MobileNetSSD_deploy.caffemodel")

# ---------------------------------------------------------------------------
# Detection constant
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD: float = 0.80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_output_dir(output_dir: str = OUTPUT_PATH) -> None:
    """Create the output directory if it does not exist."""
    os.makedirs(output_dir, exist_ok=True)


def print_separator(char: str = "-", width: int = 50) -> None:
    print(char * width)


def print_header(title: str, char: str = "=", width: int = 50) -> None:
    print(char * width)
    print(title.center(width))
    print(char * width)
