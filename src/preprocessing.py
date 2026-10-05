"""
OCR preprocessing pipeline.

Pipeline:
    RAW IMAGE → GRAYSCALE → GAUSSIAN BLUR → DESKEW → ADAPTIVE THRESHOLD
"""

import cv2
import numpy as np


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to grayscale.

    Args:
        image: BGR image array.

    Returns:
        Single-channel grayscale image.
    """
    if image.ndim == 2:
        return image  # already grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply Gaussian blur to reduce noise before OCR.

    Args:
        image: Grayscale image array.
        kernel_size: Odd integer kernel size (default 3).

    Returns:
        Blurred grayscale image.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # ensure odd
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def deskew(image: np.ndarray) -> np.ndarray:
    """
    Detect dominant text orientation and rotate the image to align text horizontally.

    Uses the minimum-area bounding rectangle of all foreground pixels to
    estimate the skew angle, then applies an affine rotation to correct it.

    Args:
        image: Grayscale image array.

    Returns:
        Deskewed grayscale image (unchanged if skew is negligible).
    """
    # Threshold to isolate foreground pixels
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(binary > 0))
    if coords.shape[0] < 10:
        # Not enough foreground pixels — return unchanged
        return image

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]  # angle in [-90, 0)

    # Convert to a rotation angle in (-45, 45]
    if angle < -45:
        angle = 90 + angle

    # Skip rotation if angle is negligible (< 0.5 degrees)
    if abs(angle) < 0.5:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return deskewed


def apply_adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """
    Apply adaptive thresholding to produce a binary image for OCR.

    Adaptive thresholding handles uneven illumination better than a global
    threshold, forcing the image toward black-and-white to improve character
    contrast.

    Args:
        image: Grayscale image array.

    Returns:
        Binary (black-and-white) image.
    """
    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )


def run_preprocessing_pipeline(
    image: np.ndarray, output_dir: str = "output"
) -> np.ndarray:
    """
    Execute the full OCR preprocessing pipeline and save intermediate images.

    Steps:
        1. Grayscale
        2. Gaussian Blur
        3. Deskew
        4. Adaptive Threshold

    Args:
        image: Original BGR image.
        output_dir: Directory to save intermediate images.

    Returns:
        Final preprocessed (thresholded) image ready for OCR.
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Step 1 — Grayscale
    gray = convert_to_grayscale(image)
    cv2.imwrite(os.path.join(output_dir, "grayscale.jpg"), gray)
    print("  [1/4] Grayscale ................. SUCCESS")

    # Step 2 — Gaussian Blur
    blurred = apply_gaussian_blur(gray)
    cv2.imwrite(os.path.join(output_dir, "blurred.jpg"), blurred)
    print("  [2/4] Gaussian Blur ............. SUCCESS")

    # Step 3 — Deskew
    deskewed = deskew(blurred)
    cv2.imwrite(os.path.join(output_dir, "deskewed.jpg"), deskewed)
    print("  [3/4] Deskew .................... SUCCESS")

    # Step 4 — Adaptive Threshold
    thresholded = apply_adaptive_threshold(deskewed)
    cv2.imwrite(os.path.join(output_dir, "thresholded.jpg"), thresholded)
    print("  [4/4] Thresholding .............. SUCCESS")

    return thresholded
