import cv2
import numpy as np

def detect_traffic_light(frame):
    output = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Convert to grayscale for Hough Circle detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # Detect circles
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
        param1=100, param2=30, minRadius=10, maxRadius=60
    )

    message = ""
    color = (255, 255, 255)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Create a circular mask
            mask = np.zeros(frame.shape[:2], dtype="uint8")
            cv2.circle(mask, (x, y), r, 255, -1)
            mean_val = cv2.mean(hsv, mask=mask)

            h, s, v = mean_val[:3]

            # Green range
            if 35 < h < 85 and s > 80 and v > 80:
                message = "GO"
                color = (0, 255, 0)
            # Red range
            elif (0 <= h <= 10 or 160 <= h <= 179) and s > 80 and v > 80:
                message = "STOP"
                color = (0, 0, 255)

            # Draw the detected circle
            cv2.circle(output, (x, y), r, color, 3)
            cv2.putText(output, message, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
            break  # Only show the first detected light

    return output

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = detect_traffic_light(frame)
    cv2.imshow("Traffic Light Detection", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
