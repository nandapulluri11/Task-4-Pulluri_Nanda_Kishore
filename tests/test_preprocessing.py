"""Tests for the preprocessing pipeline using synthetic images."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import pytest

from src.preprocessing import (
    convert_to_grayscale,
    apply_gaussian_blur,
    deskew,
    apply_adaptive_threshold,
)


def make_bgr_image(h: int = 100, w: int = 100) -> np.ndarray:
    """Create a simple synthetic BGR image."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (90, 90), (200, 200, 200), -1)
    cv2.putText(img, "TEST", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return img


def make_gray_image(h: int = 100, w: int = 100) -> np.ndarray:
    """Create a simple synthetic grayscale image."""
    return cv2.cvtColor(make_bgr_image(h, w), cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# Grayscale
# ---------------------------------------------------------------------------

def test_grayscale_output_is_2d():
    img = make_bgr_image()
    gray = convert_to_grayscale(img)
    assert gray.ndim == 2, "Grayscale image must be 2-dimensional"


def test_grayscale_already_gray():
    gray = make_gray_image()
    result = convert_to_grayscale(gray)
    assert result.ndim == 2
    np.testing.assert_array_equal(result, gray)


def test_grayscale_shape():
    img = make_bgr_image(80, 120)
    gray = convert_to_grayscale(img)
    assert gray.shape == (80, 120)


# ---------------------------------------------------------------------------
# Gaussian Blur
# ---------------------------------------------------------------------------

def test_gaussian_blur_output_shape():
    gray = make_gray_image()
    blurred = apply_gaussian_blur(gray)
    assert blurred.shape == gray.shape


def test_gaussian_blur_reduces_variance():
    gray = make_gray_image()
    blurred = apply_gaussian_blur(gray, kernel_size=5)
    assert float(np.var(blurred)) <= float(np.var(gray)) + 1e-3


def test_gaussian_blur_even_kernel_corrected():
    gray = make_gray_image()
    # Even kernel should be auto-corrected to odd
    result = apply_gaussian_blur(gray, kernel_size=4)
    assert result.shape == gray.shape


# ---------------------------------------------------------------------------
# Deskew
# ---------------------------------------------------------------------------

def test_deskew_output_shape():
    gray = make_gray_image()
    result = deskew(gray)
    assert result.shape == gray.shape


def test_deskew_no_change_on_straight_image():
    """A perfectly horizontal image should not be significantly altered."""
    img = np.ones((100, 200), dtype=np.uint8) * 255
    cv2.putText(img, "HELLO", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,), 2)
    result = deskew(img)
    assert result.shape == img.shape


# ---------------------------------------------------------------------------
# Adaptive Threshold
# ---------------------------------------------------------------------------

def test_adaptive_threshold_output_shape():
    gray = make_gray_image()
    thresh = apply_adaptive_threshold(gray)
    assert thresh.shape == gray.shape


def test_adaptive_threshold_binary_values():
    gray = make_gray_image()
    thresh = apply_adaptive_threshold(gray)
    unique_vals = np.unique(thresh)
    assert set(unique_vals).issubset({0, 255}), "Threshold output must be binary (0 or 255)"


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

def test_full_pipeline_returns_binary(tmp_path):
    from src.preprocessing import run_preprocessing_pipeline
    img = make_bgr_image(100, 200)
    result = run_preprocessing_pipeline(img, output_dir=str(tmp_path))
    assert result.ndim == 2
    unique_vals = np.unique(result)
    assert set(unique_vals).issubset({0, 255})


def test_pipeline_saves_intermediate_files(tmp_path):
    from src.preprocessing import run_preprocessing_pipeline
    img = make_bgr_image(100, 200)
    run_preprocessing_pipeline(img, output_dir=str(tmp_path))
    for fname in ("grayscale.jpg", "blurred.jpg", "deskewed.jpg", "thresholded.jpg"):
        assert (tmp_path / fname).exists(), f"Missing intermediate file: {fname}"
