# Face Recognition Attendance (Web)

A Flask + OpenCV web version of your Face Recogniser Attendance System with extra safeguards.

## Features
- Register employees and capture face samples from the browser (uses your webcam).
- Train an **LBPH** face recognizer (OpenCV contrib).
- Track attendance in real time; save a CSV for the session.
- **Safeguards**:
  - **Phone detection** (optional) blocks attendance if a phone is visible.
  - **Multi-person guard** blocks if 2+ faces are in frame.
  - **Already marked** shows when the same person is recognized again on the same day.

## How to run (Localhost)
1. Install Python 3.10+
2. (Recommended) Create and activate a virtualenv.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   python app.py
   ```
5. Open http://127.0.0.1:5000 in Chrome/Edge/Firefox and allow camera access.

> Note: LBPH recognizer requires **opencv-contrib-python** (already in requirements).
> The Haar cascade is loaded from `cv2.data.haarcascades` so no extra XML download is needed.

## Optional: Phone detection model
To enable phone detection (COCO class id **77 = cell phone**), place one of these model pairs into the `models/` folder:
- **TensorFlow**: `ssd_mobilenet_v3_large_coco.pb` + `ssd_mobilenet_v3_large_coco.pbtxt`
- **TensorFlow**: `frozen_inference_graph.pb` + `ssd_mobilenet_v3_large_coco.pbtxt`
- **Caffe**: `MobileNetSSD_deploy.caffemodel` + `MobileNetSSD_deploy.prototxt`

If the model is not present, the app still runs; only phone detection is disabled.

## Directory structure
- `TrainingImage/` — captured face crops (`name.Id.sample.jpg`)
- `TrainingImageLabel/Trainer.yml` — trained LBPH model
- `EmployeeDetails/EmployeeDetails.csv` — simple ID->Name registry
- `Attendance/Attendance_YYYY-MM-DD_HH-MM-SS.csv` — saved attendance CSVs
- `models/` — place COCO SSD model files here (optional)

## Tuning
- Confidence threshold for a positive match is **< 60** in `/api/recognize`. Lower is stricter.
- Adjust the tracking loop rate in `templates/index.html` (200 ms ≈ 5 fps).

## Production
Run behind a production WSGI server (e.g., gunicorn) and serve over HTTPS so browsers allow camera on your domain.
