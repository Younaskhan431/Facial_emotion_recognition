"""
Real-time webcam demo: detect faces with MediaPipe, classify each face's
emotion, draw the results. Run locally with:

    python src/realtime.py

Press 'q' to quit.
"""

import os
import sys

import cv2

sys.path.append(os.path.dirname(__file__))
from face_detector import FaceDetector
from emotion_model import EmotionModel

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "final_model.keras")


def main():
    detector = FaceDetector(min_confidence=0.6)
    emotion_model = EmotionModel(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera index/permissions.")

    print("Press 'q' to quit.")
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes = detector.detect(frame_rgb)

        for (x1, y1, x2, y2) in boxes:
            face_crop = frame_rgb[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            label, confidence, _ = emotion_model.predict(face_crop)

            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)
            text = f"{label} ({confidence:.0%})"
            cv2.putText(
                frame_bgr, text, (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2,
            )

        cv2.imshow("Facial Emotion Recognition", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
