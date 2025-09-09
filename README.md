# 🏠 Floor Plan Analysis System

This project provides a **Floor Plan Analyzer** built with **YOLOv8** (for symbol detection) and **PaddleOCR** (for text extraction). It enables automated detection of architectural elements such as **door symbols** and **text labels (e.g., STR, BR, LR)** in scanned or digital floor plan images.

## 🚀 Features
- **YOLOv8 Object Detection** – Detects door symbols in floor plans.
- **PaddleOCR Integration** – Extracts text labels with high accuracy.
- **Preprocessing for OCR** – Enhances image contrast, reduces noise, and applies thresholding.
- **Robust Label Matching** – Uses exact, partial, and fuzzy text matching to reduce OCR errors.
- **Visualization** – Annotated results with:
  - 🔴 Red boxes → Target text labels
  - 🔵 Blue boxes → Door symbols
  - 🟢 Green boxes → All text (debug mode)
- **Streamlit App** – Interactive UI for uploading images, adjusting parameters, and downloading results.
- **Script Mode** – CLI usage for batch automation.

---

## 🛠️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/floor-plan-analyzer.git
cd floor-plan-analyzer
pip install -r requirements.txt
```

Or install manually:

```bash
pip install paddlepaddle paddleocr ultralytics streamlit opencv-python pillow pandas matplotlib numpy
```

For GPU:
```bash
pip install paddlepaddle-gpu
```

---

## ▶️ Usage

### Run Streamlit App
```bash
streamlit run app.py
```

### Run as Script
```bash
python app.py --model best.pt --image sample_floorplan.png --label STR --confidence 0.5 --output output.png
```

---

## 📊 Training Notes

- The provided implementation is trained with **YOLOv8-s (small)** model.  
- If you train with **YOLOv8-x (extra-large)** and then apply **knowledge distillation**, you can achieve:  
  - **Smaller model size (nano-level)**  
  - **High performance (close to YOLOv8-x)**  
- This is ideal for deployment in edge devices with limited resources.

---

## 📥 Output
- Annotated image with detected doors and text labels.
- Text report summarizing detections, confidence scores, and positions.

---

## 🎯 Applications
- Automated floor plan analysis
- Smart building design
- Construction document digitization
- Indoor navigation systems
