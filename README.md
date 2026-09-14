# Industrial Surface Defect Detection System

An AI-powered computer vision system for detecting and classifying surface defects in industrial metal surfaces using YOLOv8, OpenCV, Flask, ONNX Runtime, PLC integration, and Prometheus/Grafana monitoring.

## Project Overview

The Industrial Surface Defect Detection System automatically detects surface defects from industrial metal images and video streams.

The system provides:

- Image-based defect detection
- Video-based defect detection
- Real-time webcam detection
- YOLOv8 object detection
- Bounding-box visualization
- Defect classification
- Confidence score display
- ONNX model deployment support
- PLC communication and simulation
- Real-time system monitoring
- Prometheus metrics
- Grafana monitoring dashboard
- Edge inference benchmarking
- Error analysis
- Hyperparameter tuning support
- Docker deployment configuration

---

## Detected Defect Classes

The system detects six types of industrial surface defects:

1. Crazing
2. Inclusion
3. Patches
4. Pitted Surface
5. Scratches
6. Rolled-in Scale

---

## Technologies Used

### Machine Learning
- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- ONNX
- ONNX Runtime

### Backend
- Flask
- REST APIs
- Python

### Industrial Integration
- PLC communication API
- PLC simulator
- Detection coordinates
- Conveyor/rejection signal simulation

### Monitoring
- Prometheus
- Grafana
- prometheus-client

### Deployment
- Docker
- Docker Compose
- ONNX Runtime

### Frontend
- HTML5
- CSS3
- JavaScript

---

## Dataset

The project uses the **NEU Surface Defect Dataset (NEU-DET)**.

Dataset structure:

```text
dataset/
└── NEU-DET/
    ├── IMAGES/
    └── ANNOTATIONS/