"""
download_models.py
==================
Downloads the MobileNet-SSD model files required for object detection.

Run this script once before using the application:
    python download_models.py
"""

import os
import sys
import urllib.request
import hashlib

MODELS_DIR = "models"

# The prototxt is small text — always downloadable from GitHub raw
PROTOTXT_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/dnn/MobileNetSSD_deploy.prototxt"
)
PROTOTXT_PATH = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.prototxt")

# Caffemodel mirrors (tried in order)
CAFFEMODEL_URLS = [
    # Sourceforge direct binary (no LFS)
    "https://sourceforge.net/projects/dcaffe/files/MobileNetSSD_deploy.caffemodel/download",
    # Dropbox mirror
    "https://www.dropbox.com/s/8pgf1ywjqxfqhzm/MobileNetSSD_deploy.caffemodel?dl=1",
]
CAFFEMODEL_PATH = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.caffemodel")
CAFFEMODEL_EXPECTED_SIZE = 23147564  # bytes (~22 MB)


def _is_real_binary(path: str) -> bool:
    """Return True if the file looks like a real protobuf binary (not HTML/LFS pointer)."""
    if not os.path.isfile(path):
        return False
    if os.path.getsize(path) < 1_000_000:  # less than 1 MB → definitely wrong
        return False
    with open(path, "rb") as f:
        header = f.read(10)
    # Real protobuf starts with byte 0x0a or similar non-HTML bytes
    return header[0] not in (ord("<"), ord("{"), ord("v"))


def _download(url: str, dest: str, label: str) -> bool:
    """Download url to dest. Returns True on success."""
    print(f"  Downloading {label} ...")
    print(f"  URL: {url}")
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/octet-stream",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            data = response.read()
        with open(dest, "wb") as f:
            f.write(data)
        size = os.path.getsize(dest)
        print(f"  Downloaded: {size / 1024 / 1024:.2f} MB")
        return True
    except Exception as exc:
        print(f"  Failed: {exc}")
        return False


def download_prototxt() -> bool:
    if os.path.isfile(PROTOTXT_PATH) and os.path.getsize(PROTOTXT_PATH) > 10_000:
        print(f"  prototxt already present ({os.path.getsize(PROTOTXT_PATH)} bytes) — skipping.")
        return True
    return _download(PROTOTXT_URL, PROTOTXT_PATH, "MobileNetSSD_deploy.prototxt")


def download_caffemodel() -> bool:
    if _is_real_binary(CAFFEMODEL_PATH):
        print(f"  caffemodel already present ({os.path.getsize(CAFFEMODEL_PATH) / 1024 / 1024:.1f} MB) — skipping.")
        return True

    for url in CAFFEMODEL_URLS:
        ok = _download(url, CAFFEMODEL_PATH, "MobileNetSSD_deploy.caffemodel")
        if ok and _is_real_binary(CAFFEMODEL_PATH):
            return True
        print("  Downloaded file does not appear to be a valid binary. Trying next mirror...")

    return False


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 50)
    print(" MobileNet-SSD Model Downloader")
    print("=" * 50)

    print("\n[1/2] Prototxt (architecture):")
    proto_ok = download_prototxt()

    print("\n[2/2] Caffemodel (weights):")
    model_ok = download_caffemodel()

    print("\n" + "=" * 50)
    if proto_ok and model_ok:
        print(" Download complete. You can now run:")
        print("   python app.py --image input/sample.jpg --mode detection")
    else:
        print(" Automatic download failed for one or more files.")
        print("\n Manual download instructions:")
        print("   1. Visit: https://github.com/chuanqi305/MobileNet-SSD")
        print("   2. Download MobileNetSSD_deploy.caffemodel (~22 MB)")
        print("   3. Place it in the models/ directory")
        print("\n   Prototxt URL (always works):")
        print(f"   {PROTOTXT_URL}")
    print("=" * 50)


if __name__ == "__main__":
    main()
