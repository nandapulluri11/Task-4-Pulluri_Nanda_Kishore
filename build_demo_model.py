"""
build_demo_model.py
===================
Builds a minimal but REAL working Caffe model for pipeline demonstration.

Confirmed working BlobProto format (from OpenCV caffe_importer.cpp):
  - Shape: old-style fields num=1, channels=2, height=3, width=4
  - Data:  field 5, packed float32 bytes

Run:
    python build_demo_model.py
"""

import os
import struct
import numpy as np

MODELS_DIR  = "models"
PROTO_PATH  = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.prototxt")
MODEL_PATH  = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.caffemodel")
MIN_REAL_SIZE = 5_000_000  # real model is ~22 MB


# ---------------------------------------------------------------------------
# Protobuf helpers
# ---------------------------------------------------------------------------

def varint(v: int) -> bytes:
    out = b''
    while True:
        bits = v & 0x7F; v >>= 7
        out += bytes([bits | (0x80 if v else 0)])
        if not v: break
    return out

def flen(fid: int, data: bytes) -> bytes:
    """Length-delimited field (wire type 2)."""
    return varint((fid << 3) | 2) + varint(len(data)) + data

def fint(fid: int, v: int) -> bytes:
    """Varint field (wire type 0)."""
    return varint((fid << 3) | 0) + varint(v)


def blob_proto_old(arr: np.ndarray, num: int, channels: int, height: int, width: int) -> bytes:
    """
    BlobProto using old-style shape fields (confirmed working with OpenCV):
      field 1 = num
      field 2 = channels
      field 3 = height
      field 4 = width
      field 5 = data (packed float32)
    """
    floats = arr.astype(np.float32).flatten()
    packed = struct.pack(f'<{len(floats)}f', *floats)
    blob = fint(1, num) + fint(2, channels) + fint(3, height) + fint(4, width)
    blob += flen(5, packed)
    return blob


def v1layer(name: str, bottom: str, top: str, layer_type: int, blobs: list) -> bytes:
    """
    V1LayerParameter:
      field 4 = name
      field 2 = bottom
      field 3 = top
      field 5 = type (enum)
      field 6 = blobs (repeated BlobProto)
    """
    msg = flen(4, name.encode())
    msg += flen(2, bottom.encode())
    msg += flen(3, top.encode())
    msg += fint(5, layer_type)
    for b in blobs:
        msg += flen(6, b)
    return msg


# ---------------------------------------------------------------------------
# Demo prototxt — simple conv → pool → reshape to (1,1,N,7)
# ---------------------------------------------------------------------------

DEMO_PROTOTXT = """\
name: "MobileNet-SSD-Demo"
input: "data"
input_shape { dim: 1  dim: 3  dim: 300  dim: 300 }

layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  convolution_param {
    num_output: 21
    kernel_size: 1
    pad: 0
    stride: 1
    bias_term: true
  }
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param { pool: AVE  global_pooling: true }
}
layer {
  name: "reshape1"
  type: "Reshape"
  bottom: "pool1"
  top: "detection_out"
  reshape_param { shape { dim: 1  dim: 1  dim: 3  dim: 7 } }
}
"""


def _is_real_model(path: str) -> bool:
    if not os.path.isfile(path) or os.path.getsize(path) < MIN_REAL_SIZE:
        return False
    with open(path, "rb") as f:
        h = f.read(4)
    return h[0] not in (ord("<"), ord("{"), ord("v"), ord("g"))


def build():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Write prototxt
    with open(PROTO_PATH, "w") as f:
        f.write(DEMO_PROTOTXT)
    print(f"  Wrote prototxt : {PROTO_PATH}")

    # Weights for conv1: shape (21, 3, 1, 1)
    np.random.seed(42)
    weights = (np.random.randn(21, 3, 1, 1) * 0.01).astype(np.float32)
    biases  = np.zeros(21, dtype=np.float32)

    w_blob = blob_proto_old(weights.flatten(), num=21, channels=3, height=1, width=1)
    b_blob = blob_proto_old(biases,            num=1,  channels=21, height=1, width=1)

    # V1LayerParameter type 14 = CONVOLUTION
    conv_layer = v1layer("conv1", "data", "conv1", layer_type=14, blobs=[w_blob, b_blob])

    # NetParameter: field 1=name, field 2=layers (V1LayerParameter)
    net_bytes = flen(1, b"MobileNet-SSD-Demo") + flen(2, conv_layer)

    with open(MODEL_PATH, "wb") as f:
        f.write(net_bytes)
    print(f"  Wrote caffemodel: {MODEL_PATH} ({len(net_bytes):,} bytes)")

    # Verify
    import cv2
    try:
        net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
        print("  OpenCV load    : OK")
        blob = cv2.dnn.blobFromImage(
            np.zeros((300, 300, 3), dtype=np.uint8), 0.007843, (300, 300), 127.5
        )
        net.setInput(blob)
        out = net.forward()
        print(f"  Forward pass   : OK — output shape {out.shape}")
        return True
    except cv2.error as e:
        print(f"  ERROR: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print(" MobileNet-SSD Demo Model Builder")
    print("=" * 50)

    if _is_real_model(MODEL_PATH):
        print(f"\n  Real caffemodel already present ({os.path.getsize(MODEL_PATH)/1024/1024:.1f} MB).")
        print("  No action needed.")
    else:
        print("\n  Building demo model for pipeline verification ...")
        print("  NOTE: Random weights — not real detections.")
        print("        Replace with official 22 MB caffemodel for real detection.\n")
        ok = build()
        if ok:
            print("\n  Demo model ready.")
            print("  Run: python app.py --image input/sample.jpg --mode detection")
        else:
            print("\n  Build failed.")
    print("=" * 50)
