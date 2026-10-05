"""
Object detection using MobileNet-SSD via OpenCV DNN.

Supported model formats:
    1. Caffe  — MobileNetSSD_deploy.prototxt + MobileNetSSD_deploy.caffemodel
    2. ONNX   — MobileNetSSD.onnx  (auto-downloaded if Caffe model absent)

Model files expected at:
    models/MobileNetSSD_deploy.prototxt   (text, ~44 KB)
    models/MobileNetSSD_deploy.caffemodel (binary, ~22 MB)
  OR
    models/MobileNetSSD.onnx              (binary, ~23 MB)

Download the Caffe model from:
    https://github.com/chuanqi305/MobileNet-SSD
    (MobileNetSSD_deploy.caffemodel  ~22 MB)

The prototxt is downloaded automatically by download_models.py.
"""

import os
import urllib.request
import cv2
import numpy as np
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD: float = 0.80

PROTOTXT_PATH: str  = os.path.join("models", "MobileNetSSD_deploy.prototxt")
MODEL_PATH: str     = os.path.join("models", "MobileNetSSD_deploy.caffemodel")
ONNX_PATH: str      = os.path.join("models", "MobileNetSSD.onnx")

# MobileNet-SSD PASCAL VOC class labels (index 0 = background)
CLASS_LABELS = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

BLOB_SIZE: Tuple[int, int] = (300, 300)
BLOB_SCALE: float = 0.007843   # 1/127.5
BLOB_MEAN: float  = 127.5

# ONNX mirror (OpenCV model zoo)
_ONNX_URL = (
    "https://github.com/onnx/models/raw/main/validated/"
    "vision/object_detection_segmentation/ssd-mobilenetv1/"
    "model/ssd_mobilenet_v1_10.onnx"
)


# ---------------------------------------------------------------------------
# Model file validation
# ---------------------------------------------------------------------------

def _is_valid_binary(path: str, min_size: int = 100) -> bool:
    """Return True if path exists, has content, and is not an HTML redirect."""
    if not os.path.isfile(path):
        return False
    if os.path.getsize(path) < min_size:
        return False
    with open(path, "rb") as f:
        header = f.read(4)
    # HTML pages start with '<' (0x3C); Git LFS pointers start with 'v'
    return header[0] not in (ord("<"), ord("{"), ord("v"), ord("g"))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_detection_model(
    prototxt_path: str = PROTOTXT_PATH,
    model_path: str    = MODEL_PATH,
) -> cv2.dnn_Net:
    """
    Load the MobileNet-SSD model using OpenCV DNN.

    Tries Caffe format first, then ONNX if Caffe files are unavailable.

    Args:
        prototxt_path: Path to the .prototxt architecture file.
        model_path:    Path to the .caffemodel weights file.

    Returns:
        Loaded cv2.dnn_Net object.

    Raises:
        FileNotFoundError: If no usable model files are found.
        RuntimeError:      If OpenCV fails to load the model.
    """
    # --- Try Caffe ---
    if _is_valid_binary(model_path) and os.path.isfile(prototxt_path):
        try:
            net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            return net
        except cv2.error as exc:
            raise RuntimeError(
                f"\nERROR: OpenCV failed to load the MobileNet-SSD Caffe model.\n"
                f"  Detail: {exc}\n"
                f"  The prototxt and caffemodel may be mismatched.\n"
                f"  Download both files from: https://github.com/chuanqi305/MobileNet-SSD"
            ) from exc

    # --- Try ONNX ---
    if _is_valid_binary(ONNX_PATH):
        try:
            net = cv2.dnn.readNetFromONNX(ONNX_PATH)
            return net
        except cv2.error:
            pass

    # --- Neither available ---
    missing = []
    if not _is_valid_binary(model_path):
        missing.append(f"  - {model_path}  (MobileNet-SSD weights, ~22 MB)")
    if not os.path.isfile(prototxt_path):
        missing.append(f"  - {prototxt_path}  (MobileNet-SSD architecture)")

    raise FileNotFoundError(
        f"\nERROR: MobileNet-SSD model files not found or invalid.\n\n"
        f"  Missing files:\n" + "\n".join(missing) + "\n\n"
        f"  Quick fix — build a demo model for pipeline testing:\n"
        f"    python build_demo_model.py\n\n"
        f"  For real object detection, download the official model (~22 MB):\n"
        f"    https://github.com/chuanqi305/MobileNet-SSD\n"
        f"    File: MobileNetSSD_deploy.caffemodel  →  place in models/\n"
    )


# ---------------------------------------------------------------------------
# Blob creation
# ---------------------------------------------------------------------------

def create_blob(image: np.ndarray) -> np.ndarray:
    """
    Create a 300x300 blob from an image for MobileNet-SSD inference.

    Args:
        image: BGR image array (any size).

    Returns:
        4-D blob array of shape (1, 3, 300, 300).
    """
    return cv2.dnn.blobFromImage(
        image,
        scalefactor=BLOB_SCALE,
        size=BLOB_SIZE,
        mean=BLOB_MEAN,
        swapRB=False,
        crop=False,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def detect_objects(net: cv2.dnn_Net, image: np.ndarray) -> np.ndarray:
    """
    Run forward inference through MobileNet-SSD.

    Args:
        net:   Loaded cv2.dnn_Net model.
        image: BGR image array.

    Returns:
        Raw detections array with shape (1, 1, N, 7).
    """
    blob = create_blob(image)
    net.setInput(blob)
    detections = net.forward()
    return detections


# ---------------------------------------------------------------------------
# Decode & filter detections
# ---------------------------------------------------------------------------

def decode_detections(
    detections: np.ndarray,
    image_width: int,
    image_height: int,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> List[dict]:
    """
    Decode raw MobileNet-SSD detections into structured results.

    Converts normalised bounding-box coordinates to pixel coordinates and
    filters out detections below the confidence threshold.

    Args:
        detections:           Raw output from detect_objects() — shape (1,1,N,7).
        image_width:          Width of the original image in pixels.
        image_height:         Height of the original image in pixels.
        confidence_threshold: Minimum confidence to accept a detection (default 0.80).

    Returns:
        List of accepted detection dicts with keys:
            label, confidence, start_x, start_y, end_x, end_y, accepted
    """
    accepted = []

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        class_id   = int(detections[0, 0, i, 1])

        if 0 <= class_id < len(CLASS_LABELS):
            label = CLASS_LABELS[class_id]
        else:
            label = f"class_{class_id}"

        if label == "background":
            continue

        status = "ACCEPTED" if confidence >= confidence_threshold else "REJECTED"

        # Convert normalised → pixel coordinates
        start_x = int(detections[0, 0, i, 3] * image_width)
        start_y = int(detections[0, 0, i, 4] * image_height)
        end_x   = int(detections[0, 0, i, 5] * image_width)
        end_y   = int(detections[0, 0, i, 6] * image_height)

        # Clamp to image boundaries
        start_x = max(0, min(start_x, image_width  - 1))
        start_y = max(0, min(start_y, image_height - 1))
        end_x   = max(0, min(end_x,   image_width  - 1))
        end_y   = max(0, min(end_y,   image_height - 1))

        print(f"  {label.capitalize():<16} {confidence * 100:6.2f}%  {status}")

        if confidence >= confidence_threshold:
            accepted.append({
                "label":      label,
                "confidence": confidence,
                "start_x":    start_x,
                "start_y":    start_y,
                "end_x":      end_x,
                "end_y":      end_y,
                "accepted":   True,
            })

    return accepted
