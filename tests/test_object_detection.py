"""Tests for object detection — confidence filtering and coordinate decoding."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.object_detection import decode_detections, create_blob, CLASS_LABELS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fake_detections(entries):
    """
    Build a fake detections array matching MobileNet-SSD output shape (1,1,N,7).

    entries: list of (class_id, confidence, x1, y1, x2, y2)
    """
    n = len(entries)
    det = np.zeros((1, 1, n, 7), dtype=np.float32)
    for i, (cls_id, conf, x1, y1, x2, y2) in enumerate(entries):
        det[0, 0, i] = [0, cls_id, conf, x1, y1, x2, y2]
    return det


# ---------------------------------------------------------------------------
# Confidence filtering
# ---------------------------------------------------------------------------

def test_all_accepted_above_threshold():
    entries = [
        (15, 0.95, 0.1, 0.1, 0.5, 0.5),  # person, 95%
        (7,  0.82, 0.2, 0.2, 0.6, 0.6),  # car, 82%
    ]
    det = make_fake_detections(entries)
    results = decode_detections(det, 640, 480, confidence_threshold=0.80)
    assert len(results) == 2
    for r in results:
        assert r["confidence"] >= 0.80


def test_all_rejected_below_threshold():
    entries = [
        (15, 0.50, 0.1, 0.1, 0.5, 0.5),
        (7,  0.30, 0.2, 0.2, 0.6, 0.6),
    ]
    det = make_fake_detections(entries)
    results = decode_detections(det, 640, 480, confidence_threshold=0.80)
    assert len(results) == 0


def test_mixed_threshold():
    entries = [
        (15, 0.91, 0.1, 0.1, 0.5, 0.5),  # ACCEPTED
        (7,  0.61, 0.2, 0.2, 0.6, 0.6),  # REJECTED
        (12, 0.85, 0.3, 0.3, 0.7, 0.7),  # ACCEPTED
    ]
    det = make_fake_detections(entries)
    results = decode_detections(det, 640, 480, confidence_threshold=0.80)
    assert len(results) == 2
    assert all(r["confidence"] >= 0.80 for r in results)


def test_exactly_at_threshold_accepted():
    entries = [(15, 0.80, 0.1, 0.1, 0.5, 0.5)]
    det = make_fake_detections(entries)
    results = decode_detections(det, 640, 480, confidence_threshold=0.80)
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Bounding box coordinate conversion
# ---------------------------------------------------------------------------

def test_bounding_box_pixel_conversion():
    """Normalised coords should be correctly scaled to pixel coords."""
    entries = [(15, 0.90, 0.1, 0.2, 0.6, 0.8)]
    det = make_fake_detections(entries)
    results = decode_detections(det, image_width=1000, image_height=500, confidence_threshold=0.0)
    # Filter out background
    person_results = [r for r in results if r["label"] == "person"]
    assert len(person_results) == 1
    r = person_results[0]
    assert r["start_x"] == 100   # 0.1 * 1000
    assert r["start_y"] == 100   # 0.2 * 500
    assert r["end_x"]   == 600   # 0.6 * 1000
    assert r["end_y"]   == 400   # 0.8 * 500


def test_bounding_box_clamped_to_image():
    """Out-of-bounds normalised coords must be clamped."""
    entries = [(15, 0.90, -0.1, -0.1, 1.5, 1.5)]
    det = make_fake_detections(entries)
    results = decode_detections(det, image_width=640, image_height=480, confidence_threshold=0.0)
    person_results = [r for r in results if r["label"] == "person"]
    assert len(person_results) == 1
    r = person_results[0]
    assert r["start_x"] >= 0
    assert r["start_y"] >= 0
    assert r["end_x"] <= 639
    assert r["end_y"] <= 479


# ---------------------------------------------------------------------------
# Blob creation
# ---------------------------------------------------------------------------

def test_create_blob_shape():
    import numpy as np
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    blob = create_blob(img)
    # Expected shape: (1, 3, 300, 300)
    assert blob.shape == (1, 3, 300, 300)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def test_load_image_missing_file():
    from src.image_loader import load_image
    with pytest.raises(FileNotFoundError):
        load_image("nonexistent_path/image.jpg")


def test_load_image_invalid_file(tmp_path):
    from src.image_loader import load_image
    bad_file = tmp_path / "bad.jpg"
    bad_file.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        load_image(str(bad_file))
