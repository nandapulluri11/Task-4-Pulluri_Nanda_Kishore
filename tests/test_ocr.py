"""Tests for the OCR module."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import pytest

from src.ocr import save_ocr_result


# ---------------------------------------------------------------------------
# save_ocr_result
# ---------------------------------------------------------------------------

def test_save_ocr_result_with_text(tmp_path):
    path = str(tmp_path / "ocr_result.txt")
    save_ocr_result("Hello World", path)
    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    assert "Hello World" in content


def test_save_ocr_result_empty_text(tmp_path):
    path = str(tmp_path / "ocr_result.txt")
    save_ocr_result("", path)
    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    assert "No readable text detected." in content


def test_save_ocr_result_whitespace_only(tmp_path):
    path = str(tmp_path / "ocr_result.txt")
    save_ocr_result("   \n\n  ", path)
    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    assert "No readable text detected." in content


# ---------------------------------------------------------------------------
# Tesseract availability (skip gracefully if not installed)
# ---------------------------------------------------------------------------

def test_tesseract_verify():
    from src.ocr import verify_tesseract
    try:
        version = verify_tesseract()
        assert version is not None
        print(f"\n  Tesseract version: {version}")
    except EnvironmentError as exc:
        pytest.skip(f"Tesseract not installed: {exc}")


def test_extract_text_on_synthetic_image():
    """
    Generate a white image with black text and attempt OCR.
    Skips if Tesseract is not installed.
    """
    from src.ocr import extract_text, verify_tesseract
    try:
        verify_tesseract()
    except EnvironmentError:
        pytest.skip("Tesseract not installed")

    # Create a clean white image with readable text
    img = np.ones((100, 400), dtype=np.uint8) * 255
    cv2.putText(img, "HELLO WORLD", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,), 3)

    text = extract_text(img, psm=7)
    # We just verify it returns a string without crashing
    assert isinstance(text, str)
