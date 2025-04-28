# 🚗 Driver Drowsiness and Yawning Detection

This project uses **OpenCV**, **Dlib**, and **Scipy** to monitor a driver's face in real-time and detect **drowsiness** (based on eye closure) and **yawning** (based on mouth opening).  
Visual alerts are displayed when drowsiness or yawning is detected to enhance driver safety.

---

## 🛠️ Features
- Real-time face detection
- Drowsiness detection using Eye Aspect Ratio (EAR)
- Yawning detection using Lip Distance
- Visual alerts on the camera feed
- Lightweight and fast (runs on a standard webcam)

---

## 🛒 Requirements

- Python 3.6 or above
- Libraries:
  - OpenCV
  - Dlib
  - Imutils
  - Scipy
  - Numpy
- CMake (required for dlib if not already installed)

---

## 📂 Project Structure

- project_folder/ 
- │ ├── driver_monitoring.py 
- ├── shape_predictor_68_face_landmarks/ 
- │ └── shape_predictor_68_face_landmarks.dat

---

## ⚙️ Installation Steps

1. Clone this repository or download the project files.

2. Install the required libraries:

```bash
pip install opencv-python
pip install dlib
pip install imutils
pip install scipy
pip install numpy
```

3. Download and place the shape_predictor_68_face_landmarks.dat model file inside a folder named shape_predictor_68_face_landmarks/.

Make sure the path in the script is correct:
```bash
shape_predictor = dlib.shape_predictor("..\shape_predictor_68_face_landmarks\shape_predictor_68_face_landmarks.dat")
```
---

## 🚀 How to Run

Run the Python script:
```bash
python driver_monitoring.py
```

- The webcam will open.
- The application will detect:
  - Drowsiness → shows Drowsiness Detected!
  - Yawning → shows Yawning Detected!
- Otherwise:
  - Shows Driver Awake and No Yawning
    
---
## ⚡ How It Works

- Eye Aspect Ratio (EAR) is calculated using eye landmarks.
  - If EAR < 0.25 → Drowsiness detected.
- Lip Distance is calculated using mouth landmarks.
- If distance > 35 → Yawning detected.
- Dlib's 68 facial landmarks model is used for keypoint detection.

---

## 🤝 Contribution
Feel free to fork the repository and submit pull requests to enhance the simulation.
