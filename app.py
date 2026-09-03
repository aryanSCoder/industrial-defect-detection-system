from flask import Flask, render_template, request, Response
from ultralytics import YOLO
import os
import uuid
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

MODEL_PATH = r"F:\industrial_defect_detection\models\best.pt"

model = YOLO(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


# ================= IMAGE DETECTION =================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No image selected"

    filename = f"{uuid.uuid4().hex}_{file.filename}"
    image_path = os.path.join(UPLOAD_FOLDER, filename)

    file.save(image_path)

    results = model.predict(
        source=image_path,
        conf=0.25,
        verbose=False
    )

    result = results[0]

    output_filename = f"result_{filename}"
    output_path = os.path.join(RESULT_FOLDER, output_filename)

    annotated_image = result.plot()
    cv2.imwrite(output_path, annotated_image)

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            detections.append({
                "class": model.names[class_id],
                "confidence": round(confidence * 100, 2)
            })

    return render_template(
        "index.html",
        uploaded_image=image_path,
        result_image=output_path,
        detections=detections,
        total_defects=len(detections)
    )


# ================= VIDEO DETECTION =================

@app.route("/predict_video", methods=["POST"])
def predict_video():

    if "video" not in request.files:
        return "No video uploaded"

    file = request.files["video"]

    if file.filename == "":
        return "No video selected"

    filename = f"{uuid.uuid4().hex}_{file.filename}"
    video_path = os.path.join(UPLOAD_FOLDER, filename)

    file.save(video_path)

    output_filename = (
        f"detected_{filename.rsplit('.', 1)[0]}.mp4"
    )

    output_path = os.path.join(
        RESULT_FOLDER,
        output_filename
    )

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        results = model.predict(
            frame,
            conf=0.25,
            verbose=False
        )

        annotated_frame = results[0].plot()

        out.write(annotated_frame)

    cap.release()
    out.release()

    return render_template(
        "index.html",
        result_video=output_path
    )


# ================= WEBCAM =================

def generate_webcam():

    cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()

        if not success:
            break

        results = model.predict(
            frame,
            conf=0.25,
            verbose=False
        )

        annotated_frame = results[0].plot()

        # Count detected defects
        total_defects = 0

        if results[0].boxes is not None:
            total_defects = len(results[0].boxes)

        cv2.putText(
            annotated_frame,
            f"Defects Detected: {total_defects}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        # Convert frame to JPEG
        _, buffer = cv2.imencode(
            ".jpg",
            annotated_frame
        )

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes +
            b"\r\n"
        )

    cap.release()


@app.route("/webcam")
def webcam():

    return render_template("webcam.html")


@app.route("/video_feed")
def video_feed():

    return Response(
        generate_webcam(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":

    app.run(
        debug=True,
        threaded=True
    )