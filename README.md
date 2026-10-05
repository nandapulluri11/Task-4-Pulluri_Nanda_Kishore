# Project 4 — Image & Text Recognition (Basic)

## Project Overview

A Python-based AI recognition pipeline that ingests raw visual data and
produces machine-readable output via two independent paths:

- **Path 1 — OCR**: Extracts text from images using Tesseract OCR
- **Path 2 — Object Detection**: Detects objects using MobileNet-SSD

## Objective

Build a recognition pipeline demonstrating:
- Image representation (matrix: height × width × channels)
- Preprocessing (grayscale → blur → deskew → threshold)
- OCR with configurable Page Segmentation Mode
- Object detection with bounding boxes
- Confidence filtering (≥ 80% threshold)
- Visual output saved to `output/`

## Architecture

```
                    INPUT IMAGE
                         |
                         v
                  IMAGE LOADING
                      OpenCV
                         |
                         v
                  PREPROCESSING
                         |
              +----------+----------+
              |                     |
              v                     v
          OCR PATH             OBJECT PATH
              |                     |
              v                     v
        PyTesseract           MobileNet-SSD
              |                     |
              v                     v
       Recognized Text       Object Predictions
                                    |
                                    v
                             Bounding Boxes
                                    |
                                    v
                       Confidence Filter (>= 80%)
                                    |
                                    v
                              FINAL OUTPUT
```

## Complete Pipeline

```
INPUT IMAGE
    ↓
OPENCV (load + validate)
    ↓
PREPROCESSING
    ↓
+------------------+------------------+
|                                     |
OCR PATH                    OBJECT DETECTION PATH
|                                     |
↓                                     ↓
Grayscale                        blobFromImage()
↓                                     ↓
Gaussian Blur                   MobileNet-SSD
↓                                     ↓
Deskew                          Raw Detections
↓                                     ↓
Adaptive Threshold              Decode + Filter
↓                                     ↓
PyTesseract                     Bounding Boxes
↓                                     ↓
TEXT                            OBJECTS (≥80%)
|                                     |
+------------------+------------------+
                   ↓
            FINAL OUTPUT
     (output/ directory files)
```

## Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Primary language |
| OpenCV (`cv2`) | Image loading, preprocessing, DNN inference, visualization |
| PyTesseract | Python wrapper for Tesseract OCR |
| Tesseract OCR | OCR engine executable |
| MobileNet-SSD | Pre-trained object detection model (Caffe format) |
| NumPy | Array operations |
| Pillow | Image format support for PyTesseract |

## Installation

### 1. Clone / open the project

```bash
cd "Image or Text Recognition"
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR (required for OCR path)

**Windows:**
Download and install from:
https://github.com/UB-Mannheim/tesseract/wiki

Default install path: `C:\Program Files\Tesseract-OCR\tesseract.exe`

The application auto-detects this path. If installed elsewhere, set:
```bash
set TESSERACT_CMD=C:\path\to\tesseract.exe
```

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt install tesseract-ocr
```

### 4. MobileNet-SSD Model Setup

#### Option A — Download the real model (~22 MB)

```bash
python download_models.py
```

Then manually download `MobileNetSSD_deploy.caffemodel` (~22 MB) from:
https://github.com/chuanqi305/MobileNet-SSD

Place it in `models/`.

#### Option B — Build a demo model for pipeline testing

```bash
python build_demo_model.py
```

This creates a minimal working model that demonstrates the full pipeline.
Detections will be random (untrained weights). Replace with the real model
for actual object detection.

## Project Structure

```
project-root/
│
├── app.py                  ← Main application entry point
├── requirements.txt        ← Python dependencies
├── README.md
├── .gitignore
├── download_models.py      ← Downloads model files
├── build_demo_model.py     ← Builds demo model for testing
│
├── input/
│   └── sample.jpg          ← Place your test images here
│
├── output/
│   ├── grayscale.jpg
│   ├── blurred.jpg
│   ├── deskewed.jpg
│   ├── thresholded.jpg
│   ├── ocr_result.txt
│   └── object_detection_result.jpg
│
├── models/
│   ├── MobileNetSSD_deploy.prototxt    ← Architecture (auto-downloaded)
│   └── MobileNetSSD_deploy.caffemodel  ← Weights (~22 MB, download manually)
│
├── src/
│   ├── __init__.py
│   ├── image_loader.py     ← OpenCV image loading + validation
│   ├── preprocessing.py    ← Grayscale, blur, deskew, threshold
│   ├── ocr.py              ← PyTesseract OCR integration
│   ├── object_detection.py ← MobileNet-SSD inference + confidence filter
│   ├── visualization.py    ← Bounding box drawing
│   └── utils.py            ← Shared constants and helpers
│
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py
    ├── test_ocr.py
    └── test_object_detection.py
```

## How to Run

### Default (both OCR + detection)

```bash
python app.py --image input/sample.jpg
```

### OCR only

```bash
python app.py --image input/sample.jpg --mode ocr
```

### Object detection only

```bash
python app.py --image input/sample.jpg --mode detection
```

### Both paths

```bash
python app.py --image input/sample.jpg --mode both
```

### With custom confidence threshold

```bash
python app.py --image input/sample.jpg --mode detection --confidence 0.80
```

### With custom Tesseract PSM

```bash
python app.py --image input/sample.jpg --mode ocr --psm 6
```

## OCR Usage

Tesseract Page Segmentation Modes (PSM):

| PSM | Description | Best for |
|-----|-------------|----------|
| 3 | Fully automatic | General documents |
| 6 | Single uniform block | Paragraphs, forms |
| 7 | Single text line | One-line images |
| 11 | Sparse text | Scattered text |

Default: PSM 6

## Object Detection Usage

The MobileNet-SSD model detects 20 PASCAL VOC object classes:

`aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa,
train, tvmonitor`

## Confidence Threshold

Default: **80%** (`CONFIDENCE_THRESHOLD = 0.80`)

- Detections ≥ 80% → **ACCEPTED** (drawn on output image)
- Detections < 80% → **REJECTED** (not shown)

Override at runtime:
```bash
python app.py --image input/sample.jpg --mode detection --confidence 0.75
```

## Preprocessing Explanation

| Step | Function | Purpose |
|------|----------|---------|
| Grayscale | `convert_to_grayscale()` | Reduce to single channel |
| Gaussian Blur | `apply_gaussian_blur()` | Remove noise |
| Deskew | `deskew()` | Align text horizontally |
| Adaptive Threshold | `apply_adaptive_threshold()` | Binary image for OCR |

Intermediate images are saved to `output/` for visual inspection.

## Test Cases

### Test Case 1 — OCR

Place a clear image with readable text in `input/`:

```
Student Name: Nanda Kishore
Roll Number: 12345
Project: Artificial Intelligence
```

Expected: Text printed to terminal and saved to `output/ocr_result.txt`

```bash
python app.py --image input/sample.jpg --mode ocr --psm 6
```

### Test Case 2 — Object Detection

Place an image containing common objects (person, car, dog, bottle, etc.)
in `input/`.

Expected: Objects with confidence ≥ 80% shown with bounding boxes in
`output/object_detection_result.jpg`

```bash
python app.py --image input/photo.jpg --mode detection --confidence 0.80
```

## Sample Output

```
==================================================
 PROJECT 4 — IMAGE & TEXT RECOGNITION
==================================================

  Mode       : BOTH
  Image      : input/sample.jpg
  Confidence : 80%
  PSM        : 6

Image Information:
  Width    : 800 px
  Height   : 400 px
  Channels : 3  (BGR color)

--------------------------------------------------
 OCR PIPELINE
--------------------------------------------------
  [1/4] Grayscale ................. SUCCESS
  [2/4] Gaussian Blur ............. SUCCESS
  [3/4] Deskew .................... SUCCESS
  [4/4] Thresholding .............. SUCCESS

================== OCR RESULT ==================

Recognized Text:
Student Name: Nanda Kishore
Roll Number: 12345
Project: Artificial Intelligence

  OCR result saved: output/ocr_result.txt

--------------------------------------------------
 OBJECT DETECTION PIPELINE
--------------------------------------------------
  Model loaded .................. SUCCESS
  Blob creation ................. SUCCESS
  Inference ..................... SUCCESS

  Detected objects:

  Person           91.40%  ACCEPTED
  Car              87.80%  ACCEPTED
  Dog              61.20%  REJECTED

  Detection image saved: output/object_detection_result.jpg

==================================================
 VALIDATION
==================================================

  Library Integration .......... PASS
  Preprocessing Integrity ...... PASS
  Confidence Benchmark ......... PASS
  Visual Confirmation .......... PASS

==================================================
 PROJECT 4 STATUS: COMPLETED
==================================================
```

## Validation Checklist

### Milestone 1 — Library Integration
- [x] OpenCV loads and processes images
- [x] PyTesseract wrapper installed
- [x] Tesseract executable detected (install separately)
- [x] MobileNet-SSD loads via `cv2.dnn.readNetFromCaffe()`

### Milestone 2 — Preprocessing Integrity
- [x] Grayscale conversion → `output/grayscale.jpg`
- [x] Gaussian blur → `output/blurred.jpg`
- [x] Deskewing → `output/deskewed.jpg`
- [x] Adaptive thresholding → `output/thresholded.jpg`

### Milestone 3 — Accuracy Benchmarking
- [x] Confidence threshold = 80%
- [x] Detections below 80% are rejected
- [x] Accepted detections are ≥ 80%
- [x] OCR produces readable text on suitable images (requires Tesseract)

### Milestone 4 — Visual Confirmation
- [x] OCR output saved to `output/ocr_result.txt`
- [x] Detection output saved to `output/object_detection_result.jpg`
- [x] Bounding boxes drawn on detected objects
- [x] Labels and confidence percentages visible

## Running Tests

```bash
python -m pytest tests/ -v
```

Expected: 24 passed, 2 skipped (OCR tests skip if Tesseract not installed)

## Limitations

1. **Tesseract must be installed separately** — PyTesseract is only a wrapper
2. **MobileNet-SSD caffemodel (~22 MB) must be downloaded manually** — GitHub
   LFS prevents automated download in restricted environments
3. **Demo model produces random detections** — replace with real caffemodel
   for actual object detection
4. **OCR accuracy depends on image quality** — works best on clean, high-contrast
   text images
5. **MobileNet-SSD detects only 20 PASCAL VOC classes**

## Future Improvements

1. Add YOLOv8 as an alternative detection backend
2. Add PDF input support for OCR
3. Add batch processing for multiple images
4. Add a simple web UI (Flask/Streamlit)
5. Add GPU acceleration via OpenCV CUDA backend
6. Add language selection for Tesseract (multi-language OCR)
7. Add NMS (Non-Maximum Suppression) for overlapping detections
8. Add confidence score histogram visualization
