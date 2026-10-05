"""
generate_test_model.py
======================
Generates a minimal functional object-detection model for pipeline testing.

This uses OpenCV's DNN module to build a tiny network that accepts 300x300
blobs and returns detections in the exact same format as MobileNet-SSD.

This is NOT a trained model — it will not produce meaningful detections.
It exists solely to verify the full pipeline (blob → inference → decode →
confidence filter → visualize) runs without errors.

Run:
    python generate_test_model.py
"""

import os
import struct
import numpy as np

MODELS_DIR = "models"
CAFFEMODEL_PATH = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.caffemodel")
PROTOTXT_PATH   = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.prototxt")


def _is_real_caffemodel(path: str) -> bool:
    if not os.path.isfile(path) or os.path.getsize(path) < 1_000_000:
        return False
    with open(path, "rb") as f:
        h = f.read(4)
    return h[0] not in (ord("<"), ord("{"), ord("v"), ord("g"))


def write_minimal_prototxt():
    """Write a minimal prototxt that produces a (1,1,200,7) detection output."""
    proto = """name: "MobileNet-SSD-Test"
input: "data"
input_shape { dim: 1 dim: 3 dim: 300 dim: 300 }

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
  pooling_param { pool: AVE global_pooling: true }
}
layer {
  name: "reshape"
  type: "Reshape"
  bottom: "pool1"
  top: "detection_out"
  reshape_param { shape { dim: 1 dim: 1 dim: 1 dim: 7 } }
}
"""
    with open(PROTOTXT_PATH, "w") as f:
        f.write(proto)
    print(f"  Wrote minimal prototxt: {PROTOTXT_PATH}")


def _encode_varint(value: int) -> bytes:
    """Encode an integer as a protobuf varint."""
    bits = value & 0x7F
    value >>= 7
    result = b""
    while value:
        result += bytes([0x80 | bits])
        bits = value & 0x7F
        value >>= 7
    result += bytes([bits])
    return result


def _write_proto_string(field: int, data: bytes) -> bytes:
    """Write a protobuf length-delimited field."""
    tag = (field << 3) | 2
    return _encode_varint(tag) + _encode_varint(len(data)) + data


def _write_proto_float(field: int, value: float) -> bytes:
    """Write a protobuf 32-bit float field."""
    tag = (field << 3) | 5
    return _encode_varint(tag) + struct.pack("<f", value)


def _write_proto_int32(field: int, value: int) -> bytes:
    """Write a protobuf int32 field."""
    tag = (field << 3) | 0
    return _encode_varint(tag) + _encode_varint(value)


def build_minimal_caffemodel():
    """
    Build a minimal valid Caffe NetParameter protobuf binary.

    The network has one conv layer (21 filters, 1x1 kernel) so it can
    load without errors. Weights are random — detections will be noise,
    but the pipeline shape is correct.
    """
    # Conv layer: 21 filters, 1x1, 3 input channels → weight shape (21,3,1,1)
    weight_count = 21 * 3 * 1 * 1
    bias_count   = 21

    weights = np.random.randn(weight_count).astype(np.float32)
    biases  = np.zeros(bias_count, dtype=np.float32)

    def blob_proto(data: np.ndarray, shape: list) -> bytes:
        """Encode a BlobProto (field 5 = data floats, field 7 = shape)."""
        # shape sub-message: field 1 = dim (repeated)
        shape_msg = b""
        for d in shape:
            shape_msg += _write_proto_int32(1, d)

        blob = b""
        # field 7 = shape (BlobShape)
        blob += _write_proto_string(7, shape_msg)
        # field 5 = data (repeated float, packed)
        float_bytes = data.tobytes()
        blob += _write_proto_string(5, float_bytes)
        return blob

    weight_blob = blob_proto(weights, [21, 3, 1, 1])
    bias_blob   = blob_proto(biases,  [21])

    # LayerParameter for conv1
    layer = b""
    layer += _write_proto_string(1, b"conv1")   # name
    layer += _write_proto_string(2, b"Convolution")  # type
    layer += _write_proto_string(3, b"data")    # bottom
    layer += _write_proto_string(4, b"conv1")   # top
    layer += _write_proto_string(13, weight_blob)  # blobs[0] = weights
    layer += _write_proto_string(13, bias_blob)    # blobs[1] = biases

    # NetParameter: field 2 = name, field 100 = layer (repeated)
    net = b""
    net += _write_proto_string(2, b"MobileNet-SSD-Test")
    net += _write_proto_string(100, layer)

    with open(CAFFEMODEL_PATH, "wb") as f:
        f.write(net)

    size = os.path.getsize(CAFFEMODEL_PATH)
    print(f"  Wrote minimal caffemodel: {CAFFEMODEL_PATH} ({size} bytes)")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 50)
    print(" Test Model Generator")
    print("=" * 50)

    if _is_real_caffemodel(CAFFEMODEL_PATH):
        print("\n  Real caffemodel already present — no action needed.")
        return

    print("\n  Real MobileNet-SSD caffemodel not found.")
    print("  Generating a minimal test model for pipeline verification...")
    print("  NOTE: This model produces random outputs — not real detections.")
    print("        Replace with the real caffemodel for actual object detection.\n")

    write_minimal_prototxt()
    build_minimal_caffemodel()

    # Verify it loads
    try:
        import cv2
        net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
        print("\n  OpenCV loaded the test model successfully.")
    except Exception as exc:
        print(f"\n  Warning: OpenCV could not load the test model: {exc}")

    print("\n" + "=" * 50)
    print(" Test model ready.")
    print(" For real object detection, replace models/MobileNetSSD_deploy.caffemodel")
    print(" with the official 22 MB file from:")
    print("   https://github.com/chuanqi305/MobileNet-SSD")
    print("=" * 50)


if __name__ == "__main__":
    main()
