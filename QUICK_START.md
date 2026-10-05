# Quick Start Guide — Project 4

## Installation (One-time Setup)

```bash
# 1. Navigate to project directory
cd "Image or Text Recognition"

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install Tesseract OCR (if not already installed)
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
# macOS: brew install tesseract
# Ubuntu: sudo apt install tesseract-ocr

# 4. Verify Tesseract is installed
tesseract --version
```

---

## Running the Application

### 1. Default Mode (Both OCR + Detection)
```bash
python app.py --image input/sample.jpg
```
**Output:**
- `output/ocr_result.txt` — Extracted text
- `output/object_detection_result.jpg` — Annotated image
- Intermediate preprocessing images

---

### 2. OCR Only
```bash
python app.py --image input/sample.jpg --mode ocr
```
**Best for:** Text extraction from documents, forms, images

---

### 3. Object Detection Only
```bash
python app.py --image input/sample.jpg --mode detection
```
**Best for:** Finding objects in images (person, car, dog, etc.)

---

### 4. With Custom Confidence Threshold
```bash
python app.py --image input/sample.jpg --mode detection --confidence 0.75
```
**Options:**
- `0.50` — Very permissive (many false positives)
- `0.75` — Moderate (balanced)
- `0.80` — Default (strict)
- `0.95` — Very strict (few detections)

---

### 5. With Custom Tesseract PSM (Page Segmentation Mode)
```bash
python app.py --image input/sample.jpg --mode ocr --psm 6
```
**PSM Options:**
- `3` — Fully automatic (default Tesseract)
- `6` — Single uniform block (default here) ← **Recommended**
- `7` — Single text line
- `11` — Sparse text

---

## Running Tests

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run Specific Test File
```bash
python -m pytest tests/test_preprocessing.py -v
python -m pytest tests/test_ocr.py -v
python -m pytest tests/test_object_detection.py -v
```

### Run Specific Test
```bash
python -m pytest tests/test_preprocessing.py::test_grayscale_output_is_2d -v
```

---

## Output Files

After running the application, check the `output/` directory:

| File | Purpose |
|------|---------|
| `grayscale.jpg` | Grayscale conversion |
| `blurred.jpg` | After Gaussian blur |
| `deskewed.jpg` | After deskewing |
| `thresholded.jpg` | After adaptive threshold |
| `ocr_result.txt` | Extracted text |
| `object_detection_result.jpg` | Annotated image with bounding boxes |

---

## Common Use Cases

### Extract Text from a Document
```bash
python app.py --image input/document.jpg --mode ocr --psm 6
cat output/ocr_result.txt
```

### Find Objects in a Photo
```bash
python app.py --image input/photo.jpg --mode detection --confidence 0.80
# View output/object_detection_result.jpg
```

### Process Multiple Images
```bash
for img in input/*.jpg; do
    python app.py --image "$img" --mode both
done
```

### Adjust OCR Accuracy
```bash
# Try different PSM values to find best results
python app.py --image input/sample.jpg --mode ocr --psm 3
python app.py --image input/sample.jpg --mode ocr --psm 6
python app.py --image input/sample.jpg --mode ocr --psm 7
```

---

## Troubleshooting

### Error: "Tesseract OCR executable was not found"
**Solution:**
1. Install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
2. Or set environment variable:
   ```bash
   set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
   ```

### Error: "MobileNet-SSD model files not found"
**Solution:**
```bash
# Build a demo model for testing
python build_demo_model.py

# Or download the real model
python download_models.py
```

### No objects detected (confidence < 80%)
**Solution:**
- Lower the confidence threshold: `--confidence 0.50`
- Use a real image with clear objects
- Replace demo model with official caffemodel

### Poor OCR results
**Solution:**
- Try different PSM: `--psm 3`, `--psm 7`, `--psm 11`
- Ensure image has high contrast
- Preprocess image manually before running

---

## Project Structure

```
Image or Text Recognition/
├── app.py                    ← Main entry point
├── requirements.txt          ← Dependencies
├── README.md                 ← Full documentation
├── VALIDATION_REPORT.md      ← Test results
├── QUICK_START.md            ← This file
│
├── input/                    ← Place images here
│   └── sample.jpg
│
├── output/                   ← Results saved here
│   ├── grayscale.jpg
│   ├── blurred.jpg
│   ├── deskewed.jpg
│   ├── thresholded.jpg
│   ├── ocr_result.txt
│   └── object_detection_result.jpg
│
├── models/                   ← Model files
│   ├── MobileNetSSD_deploy.prototxt
│   └── MobileNetSSD_deploy.caffemodel
│
├── src/                      ← Source code
│   ├── image_loader.py
│   ├── preprocessing.py
│   ├── ocr.py
│   ├── object_detection.py
│   ├── visualization.py
│   └── utils.py
│
└── tests/                    ← Test suite
    ├── test_preprocessing.py
    ├── test_ocr.py
    └── test_object_detection.py
```

---

## Key Features

✅ **Image Loading** — OpenCV with validation  
✅ **Preprocessing** — Grayscale → Blur → Deskew → Threshold  
✅ **OCR** — PyTesseract with configurable PSM  
✅ **Object Detection** — MobileNet-SSD with confidence filtering  
✅ **Visualization** — Bounding boxes with labels  
✅ **Testing** — 26 comprehensive tests  
✅ **CLI** — Flexible command-line interface  

---

## Performance

| Operation | Time |
|-----------|------|
| Image loading | < 100 ms |
| Preprocessing | < 500 ms |
| OCR | < 2 seconds |
| Detection | < 1 second |
| **Total** | **< 4 seconds** |

---

## Next Steps

1. ✅ Run: `python app.py --image input/sample.jpg`
2. ✅ Test: `python -m pytest tests/ -v`
3. ✅ Explore: Try different modes and parameters
4. ✅ Customize: Add your own images to `input/`
5. ✅ Extend: Modify code for your specific needs

---

**Status:** ✅ Project Complete & Validated  
**Test Results:** 26/26 PASSED  
**Ready to Use:** YES
