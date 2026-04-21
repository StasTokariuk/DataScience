import cv2
import numpy as np
import joblib

CATEGORIES = ["winter", "summer", "spring", "fall"]

def extract_features(image, bins=(8, 8, 8)):
    features = []

    h, w = image.shape[:2]
    h_half, w_half = h // 2, w // 2

    quadrants = [
        image[0:h_half, 0:w_half], image[0:h_half, w_half:w],
        image[h_half:h, 0:w_half], image[h_half:h, w_half:w]
    ]

    for quad in quadrants:
        hsv = cv2.cvtColor(quad, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        features.extend(hist.flatten())

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y))

    texture_hist = cv2.calcHist([magnitude], [0], None, [32], [0, 256])
    cv2.normalize(texture_hist, texture_hist)
    features.extend(texture_hist.flatten())

    return np.array(features)


if __name__ == "__main__":
    model = joblib.load("season_rf_model.pkl")

    cap = cv2.VideoCapture("spring_video.mp4")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    font_scale = max(0.7, frame_width / 1000.0)
    thickness = max(2, int(font_scale * 2))
    pos_x = int(frame_width * 0.03)
    pos_y = int(frame_height * 0.08)

    frame_count = 0
    current_season = "Unknown"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 15 == 0:
            small_frame = cv2.resize(frame, (256, 256))
            features = extract_features(small_frame)

            prediction_idx = model.predict([features])[0]
            current_season = CATEGORIES[prediction_idx]

        cv2.putText(frame, f"Detected: {current_season.upper()}", (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

        display_frame = cv2.resize(frame, (800, 600))
        cv2.imshow("Scene Understanding", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()