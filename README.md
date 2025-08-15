# Face-Recognition-Attendance-System

A real-time attendance management system using facial recognition technology.  
This application captures faces from a webcam, matches them against a pre-registered database, and automatically marks attendance with timestamps.

## Features
- Real-time face detection and recognition using OpenCV and face_recognition library.
- Automatic attendance marking with date and time.
- Prevents duplicate entries for the same person in a single session.
- Web-based interface for easy use (Flask/Django integration possible).
- Export attendance logs as CSV or Excel.
- Error handling for multiple faces or unauthorized persons.

## Tech Stack
- Programming Language: Python 3.x
- Libraries:
  - OpenCV
  - face_recognition
  - NumPy
  - Pandas
  - Flask/Django (optional for web interface)
- Database: CSV / SQLite / MySQL (configurable)
- Hardware: Webcam or external camera

## Installation & Setup

## 1. Clone the Repository:

git clone https://github.com/<your-username>/face-recognition-attendance.git
cd face-recognition-attendance


## 2. Install Dependencies:

pip install -r requirements.txt
Prepare the Dataset

## 3. Add face images to the dataset/ folder :

python train.py
This will generate encodings for the faces.

## 4. Run the Application:

python main.py


Press q to quit the camera feed.

## Usage
Start the application.

The camera will open and detect faces.

If a face matches a registered person:

Attendance is marked in attendance/Attendance-<date>.csv.

If face not recognized:

The system will ignore or alert based on configuration.

<img width="1894" height="919" alt="Screenshot 2025-08-15 135119" src="https://github.com/user-attachments/assets/a68579ba-1913-4f1a-812d-e9e7e515e3b1" />
