"""Image loading and validation using OpenCV."""

import os
import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk using OpenCV.

    Args:
        image_path: Absolute or relative path to the image file.

    Returns:
        BGR image as a NumPy array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If OpenCV cannot decode the file as an image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"ERROR: Image file not found.\n"
            f"  Expected: {image_path}\n"
            f"  Please place a valid image in the 'input/' directory."
        )

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"ERROR: Could not read image file.\n"
            f"  Path: {image_path}\n"
            f"  The file may be corrupted or in an unsupported format.\n"
            f"  Supported formats: JPG, PNG, BMP, TIFF, WEBP"
        )

    return image


def print_image_info(image: np.ndarray, image_path: str) -> None:
    """Print image matrix information (height, width, channels)."""
    if image.ndim == 3:
        height, width, channels = image.shape
    else:
        height, width = image.shape
        channels = 1

    print(f"\nImage Information:")
    print(f"  Path     : {image_path}")
    print(f"  Width    : {width} px")
    print(f"  Height   : {height} px")
    print(f"  Channels : {channels}  ({'BGR color' if channels == 3 else 'Grayscale'})")
    print(f"  Shape    : {image.shape}")
    print(f"  Dtype    : {image.dtype}")
