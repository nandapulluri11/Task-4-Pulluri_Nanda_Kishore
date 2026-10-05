"""
OCR module using PyTesseract / Tesseract OCR.

Tesseract detection order:
    1. TESSERACT_CMD environment variable
    2. Common Windows installation paths
    3. System PATH (Linux / macOS)
"""

import os
import shutil
import pytesseract
from PIL import Image
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Tesseract configuration
# ---------------------------------------------------------------------------

def _configure_tesseract() -> None:
    """
    Detect and configure the Tesseract executable path.

    Raises:
        EnvironmentError: If Tesseract cannot be found.
    """
    # 1. Honour explicit environment variable
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd and os.path.isfile(env_cmd):
        pytesseract.pytesseract.tesseract_cmd = env_cmd
        return

    # 2. Check system PATH (works on Linux / macOS / Windows if on PATH)
    if shutil.which("tesseract"):
        pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")
        return

    # 3. Common Windows installation paths
    windows_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(
            os.environ.get("USERNAME", "")
        ),
    ]
    for path in windows_paths:
        if os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return

    raise EnvironmentError(
        "\nERROR: Tesseract OCR executable was not found.\n\n"
        "  To fix this, choose one of the following options:\n\n"
        "  Option 1 — Install Tesseract:\n"
        "    Windows : https://github.com/UB-Mannheim/tesseract/wiki\n"
        "    macOS   : brew install tesseract\n"
        "    Ubuntu  : sudo apt install tesseract-ocr\n\n"
        "  Option 2 — Set the TESSERACT_CMD environment variable:\n"
        "    set TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe\n"
    )


def verify_tesseract() -> str:
    """
    Verify Tesseract is installed and return its version string.

    Returns:
        Tesseract version string.

    Raises:
        EnvironmentError: If Tesseract is not found or not functional.
    """
    _configure_tesseract()
    try:
        version = pytesseract.get_tesseract_version()
        return str(version)
    except Exception as exc:
        raise EnvironmentError(
            f"\nERROR: Tesseract was found but could not be executed.\n"
            f"  Detail: {exc}\n"
            f"  Please verify your Tesseract installation."
        ) from exc


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def extract_text(image: np.ndarray, psm: int = 6) -> str:
    """
    Run Tesseract OCR on a preprocessed image.

    Page Segmentation Modes (PSM):
        3  — Fully automatic page segmentation (default Tesseract)
        6  — Assume a single uniform block of text  ← default here
        7  — Treat the image as a single text line
        11 — Sparse text; find as much text as possible

    Args:
        image: Grayscale or binary image (NumPy array).
        psm: Tesseract Page Segmentation Mode (default 6).

    Returns:
        Recognised text string (stripped of leading/trailing whitespace).
        Returns empty string if nothing is recognised.
    """
    _configure_tesseract()

    config = f"--oem 3 --psm {psm}"

    # Convert NumPy array → PIL Image for pytesseract
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image

    try:
        raw_text = pytesseract.image_to_string(pil_image, config=config)
    except pytesseract.TesseractError as exc:
        raise RuntimeError(
            f"\nERROR: Tesseract OCR failed during recognition.\n"
            f"  Detail: {exc}"
        ) from exc

    # Clean up excessive blank lines while preserving paragraph structure
    lines = raw_text.splitlines()
    cleaned_lines = [line.rstrip() for line in lines]
    # Remove runs of more than one consecutive blank line
    result_lines = []
    prev_blank = False
    for line in cleaned_lines:
        is_blank = line.strip() == ""
        if is_blank and prev_blank:
            continue
        result_lines.append(line)
        prev_blank = is_blank

    return "\n".join(result_lines).strip()


def save_ocr_result(text: str, output_path: str) -> None:
    """
    Save OCR result text to a file.

    Args:
        text: Recognised text (may be empty).
        output_path: Destination file path.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    content = text if text.strip() else "No readable text detected."
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(content)
