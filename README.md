# Industrial Surface Defect Detection System

## Project Overview

The Industrial Surface Defect Detection System is an AI-powered computer vision application designed to automatically identify and detect surface defects in industrial materials.

The system uses a custom-trained YOLO object detection model and provides multiple methods for defect detection, including image detection, video detection, and real-time webcam detection.

---

## Features

- Custom-trained YOLO model for industrial defect detection
- Image-based defect detection
- Video-based defect detection
- Real-time webcam detection
- Defect classification
- Confidence score display
- Bounding box visualization
- Flask-based web application
- ONNX model export for deployment compatibility

---

## Technologies Used

- Python
- YOLO (Ultralytics)
- Flask
- OpenCV
- PyTorch
- ONNX
- HTML
- CSS

---

## Dataset

The project uses the NEU Surface Defect Dataset for training and testing the object detection model.

The model detects the following industrial surface defects:

- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches

---

## Project Structure

```text
industrial_defect_detection/
│
├── models/
│   ├── best.pt
│   └── best.onnx
│
├── static/
│   ├── uploads/
│   └── results/
│
├── templates/
│   ├── index.html
│   └── webcam.html
│
├── app.py
├── train.py
├── predict.py
├── webcam.py
├── create_test_video.py
├── requirements.txt
└── README.md