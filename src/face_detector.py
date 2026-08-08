"""
Thin wrapper around MediaPipe's Face Detection so realtime.py and app.py
don't have to deal with MediaPipe's API directly.
"""

import mediapipe as mp
import numpy as np


class FaceDetector:
    def __init__(self, min_confidence: float = 0.5):
        self._mp_face = mp.solutions.face_detection
        self.detector = self._mp_face.FaceDetection(
            model_selection=1,  # 1 = full-range model, better for varied distances
            min_detection_confidence=min_confidence,
        )

    def detect(self, frame_rgb: np.ndarray):
        """frame_rgb: RGB numpy array (H, W, 3).
        Returns a list of (x1, y1, x2, y2) pixel boxes, clamped to frame bounds."""
        h, w = frame_rgb.shape[:2]
        results = self.detector.process(frame_rgb)
        boxes = []
        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x1 = max(0, int(bbox.xmin * w))
                y1 = max(0, int(bbox.ymin * h))
                x2 = min(w, int((bbox.xmin + bbox.width) * w))
                y2 = min(h, int((bbox.ymin + bbox.height) * h))
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))
        return boxes

    def close(self):
        self.detector.close()
