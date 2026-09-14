from flask import Flask, jsonify, request
from datetime import datetime
import random
import threading
import time

app = Flask(__name__)

# ============================================================
# PLC STATE
# ============================================================

plc_state = {
    "connected": True,
    "machine_status": "RUNNING",
    "conveyor_status": "RUNNING",
    "inspection_status": "IDLE",
    "defect_detected": False,
    "defect_type": None,
    "confidence": 0.0,
    "x_coordinate": None,
    "y_coordinate": None,
    "rejection_signal": False,
    "last_update": None
}


# ============================================================
# UPDATE PLC
# ============================================================

def update_plc(
    defect_detected=False,
    defect_type=None,
    confidence=0.0,
    x_coordinate=None,
    y_coordinate=None
):

    plc_state["inspection_status"] = "INSPECTING"
    plc_state["defect_detected"] = defect_detected
    plc_state["defect_type"] = defect_type
    plc_state["confidence"] = round(confidence, 2)
    plc_state["x_coordinate"] = x_coordinate
    plc_state["y_coordinate"] = y_coordinate
    plc_state["last_update"] = datetime.now().isoformat()

    if defect_detected:

        plc_state["rejection_signal"] = True
        plc_state["conveyor_status"] = "STOPPED"

        print("\n" + "=" * 60)
        print("PLC DEFECT SIGNAL")
        print("=" * 60)

        print("DEFECT DETECTED")
        print("Type       :", defect_type)
        print("Confidence :", f"{confidence:.2f}%")
        print("X Position :", x_coordinate)
        print("Y Position :", y_coordinate)
        print("Action     : REJECT / STOP CONVEYOR")

    else:

        plc_state["rejection_signal"] = False
        plc_state["conveyor_status"] = "RUNNING"

        print("\nPLC STATUS: PRODUCT PASSED")


# ============================================================
# PLC STATUS
# ============================================================

@app.route("/plc/status", methods=["GET"])
def plc_status():

    return jsonify({
        "success": True,
        "plc": plc_state
    })


# ============================================================
# SEND DETECTION TO PLC
# ============================================================

@app.route("/plc/detection", methods=["POST"])
def plc_detection():

    data = request.get_json()

    if not data:
        return jsonify({
            "success": False,
            "error": "No detection data received"
        }), 400

    defect_detected = bool(
        data.get("defect_detected", False)
    )

    defect_type = data.get(
        "defect_type"
    )

    confidence = float(
        data.get("confidence", 0)
    )

    x_coordinate = data.get(
        "x_coordinate"
    )

    y_coordinate = data.get(
        "y_coordinate"
    )

    update_plc(
        defect_detected,
        defect_type,
        confidence,
        x_coordinate,
        y_coordinate
    )

    return jsonify({
        "success": True,
        "message": "Detection successfully sent to PLC simulator",
        "plc_state": plc_state
    })


# ============================================================
# RESET PLC
# ============================================================

@app.route("/plc/reset", methods=["POST"])
def reset_plc():

    plc_state["inspection_status"] = "IDLE"
    plc_state["defect_detected"] = False
    plc_state["defect_type"] = None
    plc_state["confidence"] = 0.0
    plc_state["x_coordinate"] = None
    plc_state["y_coordinate"] = None
    plc_state["rejection_signal"] = False
    plc_state["conveyor_status"] = "RUNNING"
    plc_state["last_update"] = datetime.now().isoformat()

    print("\nPLC RESET")
    print("Conveyor: RUNNING")

    return jsonify({
        "success": True,
        "message": "PLC reset successfully",
        "plc_state": plc_state
    })


# ============================================================
# SIMULATE RANDOM INSPECTION
# ============================================================

@app.route("/plc/simulate", methods=["POST"])
def simulate():

    defect_types = [
        "crazing",
        "patches",
        "inclusion",
        "pitted_surface",
        "scratches",
        "rolled-in_scale"
    ]

    defect_detected = random.choice(
        [True, True, False]
    )

    if defect_detected:

        defect_type = random.choice(
            defect_types
        )

        confidence = random.uniform(
            60,
            98
        )

        x_coordinate = random.randint(
            0,
            1000
        )

        y_coordinate = random.randint(
            0,
            1000
        )

    else:

        defect_type = None
        confidence = 0
        x_coordinate = None
        y_coordinate = None

    update_plc(
        defect_detected,
        defect_type,
        confidence,
        x_coordinate,
        y_coordinate
    )

    return jsonify({
        "success": True,
        "simulated_detection": {
            "defect_detected": defect_detected,
            "defect_type": defect_type,
            "confidence": round(
                confidence,
                2
            ),
            "x_coordinate": x_coordinate,
            "y_coordinate": y_coordinate
        },
        "plc_state": plc_state
    })


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return jsonify({
        "project": "Real-Time Industrial Defect Detection System",
        "module": "PLC Simulator",
        "status": "ONLINE",
        "endpoints": [
            "/plc/status",
            "/plc/detection",
            "/plc/reset",
            "/plc/simulate"
        ]
    })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("INDUSTRIAL PLC SIMULATOR")
    print("=" * 60)

    print("\nPLC Status : ONLINE")
    print("Server     : http://127.0.0.1:5001")

    print("\nAvailable endpoints:")
    print("GET  /plc/status")
    print("POST /plc/detection")
    print("POST /plc/reset")
    print("POST /plc/simulate")

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True
    )