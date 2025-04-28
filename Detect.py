import cv2
import dlib
import numpy as np
import time
from scipy.spatial import distance as dist
from imutils import face_utils

# Initialize camera
cap = cv2.VideoCapture(0)  # Use 0 for default camera or 1 for external camera

# Dlib face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("..\shape_predictor_68_face_landmarks\shape_predictor_68_face_landmarks.dat")

# Threshold values
eye_thresh = 0.25
yawn_thresh = 35

def detect_eye(eye):
    """Calculate eye aspect ratio to detect drowsiness."""
    poi_a = dist.euclidean(eye[1], eye[5])
    poi_b = dist.euclidean(eye[2], eye[4])
    poi_c = dist.euclidean(eye[0], eye[3])
    aspect_ratio_eye = (poi_a + poi_b) / (2.0 * poi_c)
    return aspect_ratio_eye

def cal_yawn(shape):
    """Calculate lip distance to detect yawning."""
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    distance = dist.euclidean(np.mean(top_lip, axis=0), np.mean(low_lip, axis=0))
    return distance

# Main loop for video stream
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray_frame.astype('uint8'))

    drowsiness_detected = False
    yawning_detected = False

    for face in faces:
        landmarks = shape_predictor(gray_frame, face)

        left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
        right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]

        eye_ratio = (detect_eye(left_eye) + detect_eye(right_eye)) / 2.0

        if eye_ratio < eye_thresh:
            drowsiness_detected = True

        lip_dist = cal_yawn(face_utils.shape_to_np(landmarks))
        if lip_dist > yawn_thresh:
            yawning_detected = True

        # Draw rectangle around face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        color = (0, 0, 255) if drowsiness_detected or yawning_detected else (0, 255, 0)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Display alerts
        if drowsiness_detected:
            cv2.putText(frame, "Drowsiness Detected!", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Driver Awake", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if yawning_detected:
            cv2.putText(frame, "Yawning Detected!", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No Yawning", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Driver Monitoring", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
