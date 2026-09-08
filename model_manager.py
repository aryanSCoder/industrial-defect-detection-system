from ultralytics import YOLO
import os


MODEL_FOLDER = "models"

TENSORRT_MODEL = os.path.join(
    MODEL_FOLDER,
    "best.engine"
)

ONNX_MODEL = os.path.join(
    MODEL_FOLDER,
    "best.onnx"
)

PYTORCH_MODEL = os.path.join(
    MODEL_FOLDER,
    "best.pt"
)


def load_best_available_model():

    if os.path.exists(TENSORRT_MODEL):

        print("TensorRT model found.")
        print("Loading TensorRT optimized model...")

        return YOLO(TENSORRT_MODEL), "TensorRT"

    elif os.path.exists(ONNX_MODEL):

        print("TensorRT model not available.")
        print("Loading ONNX optimized model...")

        return YOLO(ONNX_MODEL), "ONNX"

    elif os.path.exists(PYTORCH_MODEL):

        print("ONNX model not available.")
        print("Loading PyTorch model...")

        return YOLO(PYTORCH_MODEL), "PyTorch"

    else:

        raise FileNotFoundError(
            "No trained model found in the models folder."
        )