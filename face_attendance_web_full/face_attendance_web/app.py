import os
import io
import base64
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import pandas as pd

app = Flask(__name__)

# Ensure directories exist
DATA_DIRS = ["TrainingImage", "TrainingImageLabel", "EmployeeDetails", "Attendance", "models"]
for d in DATA_DIRS:
    os.makedirs(d, exist_ok=True)

EMP_CSV = os.path.join("EmployeeDetails", "EmployeeDetails.csv")
MODEL_PATH = os.path.join("TrainingImageLabel", "Trainer.yml")

# Load cascade from OpenCV's built-in data path (no local XML needed)
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Optional phone detector (COCO SSD MobileNet). Enabled only if model files are present.
PHONE_DETECTOR = {
    "net": None,
    "detector": None,
    "loaded": False,
    # COCO class id for "cell phone" is 77
    "cell_phone_id": 77
}

def try_load_phone_detector():
    # Common file names. Any one valid TF graph + pbtxt pair enables detection.
    candidates = [
        # TensorFlow models
        ("models/ssd_mobilenet_v3_large_coco.pb", "models/ssd_mobilenet_v3_large_coco.pbtxt"),
        ("models/frozen_inference_graph.pb", "models/ssd_mobilenet_v3_large_coco.pbtxt"),
        # Caffe models (kept for flexibility if a suitable prototxt is provided)
        ("models/MobileNetSSD_deploy.caffemodel", "models/MobileNetSSD_deploy.prototxt")
    ]
    for weights, config in candidates:
        if os.path.exists(weights) and os.path.exists(config):
            try:
                # Choose loader based on extension
                if weights.endswith(".pb"):
                    net = cv2.dnn.readNetFromTensorflow(weights, config)
                elif weights.endswith(".caffemodel"):
                    net = cv2.dnn.readNetFromCaffe(config, weights)
                else:
                    continue
                detector = cv2.dnn_DetectionModel(net)
                detector.setInputSize(320, 320)
                detector.setInputScale(1.0/127.5)
                detector.setInputMean((127.5, 127.5, 127.5))
                detector.setInputSwapRB(True)
                PHONE_DETECTOR["net"] = net
                PHONE_DETECTOR["detector"] = detector
                PHONE_DETECTOR["loaded"] = True
                return
            except Exception:
                continue

try_load_phone_detector()

# In-memory attendance for the current session
attendance = pd.DataFrame(columns=["Id", "Name", "Date", "Time"])

def read_employees_df():
    if os.path.exists(EMP_CSV) and os.path.getsize(EMP_CSV) > 0:
        return pd.read_csv(EMP_CSV, header=None, names=["Id", "Name"], dtype={"Id": int, "Name": str})
    return pd.DataFrame(columns=["Id", "Name"])

def upsert_employee(emp_id: int, name: str):
    df = read_employees_df()
    if (df["Id"] == emp_id).any():
        # Update name if different
        df.loc[df["Id"] == emp_id, "Name"] = name
    else:
        df.loc[len(df)] = [emp_id, name]
    df.to_csv(EMP_CSV, header=False, index=False)

def decode_base64_image(data_url: str):
    """
    Accepts a data URL like 'data:image/png;base64,....' and returns a BGR OpenCV image.
    """
    if "," in data_url:
        data = data_url.split(",", 1)[1]
    else:
        data = data_url
    img_bytes = base64.b64decode(data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_images_and_labels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(".jpg")]
    faces, Ids = [], []
    for imagePath in imagePaths:
        try:
            parts = os.path.basename(imagePath).split(".")
            # Expected: name.Id.sample.jpg
            id_from_name = int(parts[1])
            img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            Ids.append(id_from_name)
        except Exception:
            # Skip malformed filenames
            continue
    return faces, np.array(Ids, dtype=np.int32)

def ensure_recognizer():
    # Uses LBPH recognizer from opencv-contrib
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if os.path.exists(MODEL_PATH):
        recognizer.read(MODEL_PATH)
        return recognizer, True
    return recognizer, False

def detect_cell_phone(bgr_frame, conf_thresh=0.5):
    if not PHONE_DETECTOR["loaded"]:
        return False, []
    detector = PHONE_DETECTOR["detector"]
    classIds, confidences, boxes = detector.detect(bgr_frame, confThreshold=conf_thresh, nmsThreshold=0.4)
    if classIds is None or len(classIds) == 0:
        return False, []
    hits = []
    for cid, conf, box in zip(classIds.flatten(), confidences.flatten(), boxes):
        if int(cid) == PHONE_DETECTOR["cell_phone_id"] and float(conf) >= conf_thresh:
            hits.append((int(cid), float(conf), box))
    return (len(hits) > 0), hits

@app.route("/")
def index():
    return render_template("index.html", phone_model_loaded=PHONE_DETECTOR["loaded"])

@app.post("/api/capture")
def api_capture():
    """
    Save cropped face images from frames for a given employee.
    Params: id (int), name (str), image (dataURL)
    Returns: {saved: int, faces_found: int}
    """
    emp_id = request.form.get("id", "").strip()
    name = request.form.get("name", "").strip()
    image_data = request.form.get("image", "")

    # Validate
    try:
        emp_id_int = int(emp_id)
    except Exception:
        return jsonify({"ok": False, "error": "Invalid employee ID"}), 400
    if not name or not name.replace(" ", "").isalpha():
        return jsonify({"ok": False, "error": "Invalid name"}), 400
    if not image_data:
        return jsonify({"ok": False, "error": "Missing frame image"}), 400

    # Decode and basic guards
    bgr = decode_base64_image(image_data)
    if bgr is None:
        return jsonify({"ok": False, "error": "Failed to decode image"}), 400

    # Phone guard during capture
    phone_found, _ = detect_cell_phone(bgr)
    if phone_found:
        return jsonify({"ok": False, "error_code": "PHONE_DETECTED", "error": "Phone detected in frame. Remove it and try again."}), 400

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Multi-person guard
    if len(faces) >= 2:
        return jsonify({"ok": False, "error_code": "MULTIPLE_FACES", "error": "Multiple people detected. Only one person should be in frame."}), 400

    # Register/update employee (only after guards pass)
    upsert_employee(emp_id_int, name)

    saved = 0
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
        existing = [f for f in os.listdir("TrainingImage") if f".{emp_id_int}." in f]
        sample_num = len(existing) + saved + 1
        filename = f"{name}.{emp_id_int}.{sample_num}.jpg"
        cv2.imwrite(os.path.join("TrainingImage", filename), roi)
        saved += 1

    return jsonify({"ok": True, "saved": saved, "faces_found": int(len(faces))})

@app.post("/api/train")
def api_train():
    try:
        faces, Ids = get_images_and_labels("TrainingImage")
        if len(faces) == 0:
            return jsonify({"ok": False, "error": "No training images found. Capture images first."}), 400
        recognizer, _ = ensure_recognizer()
        recognizer.train(faces, Ids)
        recognizer.save(MODEL_PATH)
        return jsonify({"ok": True, "message": "Model trained successfully."})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/api/recognize")
def api_recognize():
    """
    Recognize faces from a frame and update attendance in memory.
    Params: image (dataURL)
    Returns on success:
      {ok: True, faces: [{x,y,w,h,label,conf,already}], attendance_size: int}
    Returns on guarded error:
      {ok: False, error_code: "...", error: "..."}
    """
    image_data = request.form.get("image", "")
    if not image_data:
        return jsonify({"ok": False, "error": "Missing frame image"}), 400

    bgr = decode_base64_image(image_data)
    if bgr is None:
        return jsonify({"ok": False, "error": "Failed to decode image"}), 400

    # Guard 1: phone detected
    phone_found, phone_boxes = detect_cell_phone(bgr)
    if phone_found:
        return jsonify({"ok": False, "error_code": "PHONE_DETECTED", "error": "Phone detected in frame. Remove it to continue."}), 400

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    detections = face_cascade.detectMultiScale(gray, 1.2, 5)

    # Guard 2: multi-person
    if len(detections) >= 2:
        return jsonify({"ok": False, "error_code": "MULTIPLE_FACES", "error": "Multiple people detected. Only one person should be in frame."}), 400

    recognizer, has_model = ensure_recognizer()
    employees = read_employees_df()

    faces_out = []
    global attendance

    for (x, y, w, h) in detections:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
        label = "Unknown"
        conf = None
        already_marked = False
        if has_model:
            try:
                pred_id, confidence = recognizer.predict(roi)
                conf = float(confidence)
                # Lower confidence is better for LBPH; consider < 60 as valid by default
                if confidence < 60:
                    # Map ID to name
                    row = employees[employees["Id"] == pred_id]
                    if not row.empty:
                        name = row.iloc[0]["Name"]
                        label = f"{pred_id} - {name}"
                        # Mark attendance (dedupe by Id/date)
                        ts = datetime.now()
                        date = ts.strftime("%Y-%m-%d")
                        time_str = ts.strftime("%H:%M:%S")
                        already_marked = ((attendance["Id"] == pred_id) & (attendance["Date"] == date)).any()
                        if not already_marked:
                            attendance.loc[len(attendance)] = [pred_id, name, date, time_str]
                        else:
                            label = f"{pred_id} - {name} (already marked)"
                    else:
                        label = f"{pred_id} - Unknown"
                else:
                    label = "Unknown"
            except Exception:
                label = "Unknown"
        else:
            label = "Model not trained"

        faces_out.append({
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "label": label, "conf": conf if conf is not None else None,
            "already": bool(already_marked)
        })

    return jsonify({"ok": True, "faces": faces_out, "attendance_size": int(len(attendance))})

@app.post("/api/save_attendance")
def api_save_attendance():
    global attendance
    if attendance.empty:
        return jsonify({"ok": False, "error": "No attendance to save"}), 400
    filename = f"Attendance/Attendance_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    attendance_sorted = attendance.drop_duplicates(subset=["Id", "Date"]).sort_values(by=["Date", "Time"])
    attendance_sorted.to_csv(filename, index=False)

    # Return the CSV as a file for download
    with open(filename, "rb") as f:
        data = f.read()
    mem = io.BytesIO(data)
    mem.seek(0)
    # Reset in-memory attendance for next session
    attendance = pd.DataFrame(columns=["Id", "Name", "Date", "Time"])
    return send_file(mem, mimetype="text/csv", as_attachment=True, download_name=os.path.basename(filename))

@app.get("/api/employees")
def api_employees():
    df = read_employees_df()
    return jsonify({"ok": True, "employees": df.to_dict(orient="records")})

if __name__ == "__main__":
    # For dev only. In production, run with a WSGI server.
    app.run(host="127.0.0.1", port=5000, debug=True)
