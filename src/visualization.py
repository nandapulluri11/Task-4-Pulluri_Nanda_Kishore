"""
Visualization module — draws bounding boxes and labels on detection results.
"""

import os
import cv2
import numpy as np
from typing import List


# Colour palette for bounding boxes (BGR)
_COLOURS = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 0, 255),    # red
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 165, 0),  # orange
]


def draw_detections(image: np.ndarray, detections: List[dict]) -> np.ndarray:
    """
    Draw bounding boxes and confidence labels on the image.

    Args:
        image: Original BGR image.
        detections: List of accepted detection dicts from decode_detections().

    Returns:
        Annotated BGR image copy.
    """
    annotated = image.copy()

    for idx, det in enumerate(detections):
        colour = _COLOURS[idx % len(_COLOURS)]
        label_text = f"{det['label'].capitalize()}: {det['confidence'] * 100:.1f}%"

        # Draw bounding box
        cv2.rectangle(
            annotated,
            (det["start_x"], det["start_y"]),
            (det["end_x"], det["end_y"]),
            colour,
            thickness=2,
        )

        # Draw label background
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        label_y = max(det["start_y"] - 5, text_h + 5)
        cv2.rectangle(
            annotated,
            (det["start_x"], label_y - text_h - baseline),
            (det["start_x"] + text_w, label_y + baseline),
            colour,
            thickness=cv2.FILLED,
        )

        # Draw label text
        cv2.putText(
            annotated,
            label_text,
            (det["start_x"], label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    return annotated


def save_detection_result(
    annotated_image: np.ndarray,
    output_path: str,
) -> None:
    """
    Save the annotated detection image to disk.

    Args:
        annotated_image: BGR image with drawn bounding boxes.
        output_path: Destination file path.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    cv2.imwrite(output_path, annotated_image)
