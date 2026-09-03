# 🔍 Industrial Surface Defect Detection System

An AI-powered Industrial Surface Defect Detection System developed using **YOLO, Python, Flask, and OpenCV**.

The system detects different types of surface defects from images, videos, and live webcam streams through a web-based interface.

---

## 📌 Project Overview

Quality inspection is an important part of industrial manufacturing. Manual inspection can be slow, repetitive, and prone to human error.

This project uses Artificial Intelligence and Computer Vision to automatically detect surface defects in industrial materials.

Users can:

- Upload an image for defect detection
- Upload a video for defect detection
- Use live webcam detection
- View detected defects with confidence scores
- View bounding boxes around detected defects

---

## ✨ Features

### 🖼️ Image Detection

Users can upload an image of an industrial surface. The trained YOLO model detects defects and displays:

- Original image
- Detection result
- Defect name
- Confidence score
- Total number of detected defects

### 🎥 Video Detection

Users can upload a video containing industrial surfaces.

The system processes the video frame by frame and generates a new video containing detected defects.

### 📷 Live Webcam Detection

The system supports real-time defect detection using a webcam.

The YOLO model processes live video frames and displays detected defects with bounding boxes.

### 🤖 YOLO-Based Detection

The project uses a trained YOLO object detection model for identifying industrial surface defects.

---

## 🛠️ Technologies Used

- Python
- Flask
- Ultralytics YOLO
- OpenCV
- PyTorch
- NumPy
- HTML
- CSS

---

## 📂 Project Structure

```text
industrial_defect_detection/
│
├── app.py
├── train.py
├── create_test_video.py
├── test_defect_video.mp4
├── requirements.txt
├── README.md
│
├── models/
│   └── best.pt
│
├── dataset/
│   ├── NEU-DET/
│   └── NEU-YOLO/
│
├── static/
│   ├── uploads/
│   └── results/
│
└── templates/
    ├── index.html
    └── webcam.html