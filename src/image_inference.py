"""
Run face detection + emotion recognition on a single image file and save
an annotated copy.
 
Usage:
    python src/image_inference.py --input path/to/photo.jpg --output path/to/result.jpg
"""
 
import argparse
import os
import sys
 
import cv2
 
sys.path.append(os.path.dirname(__file__))
from face_detector import FaceDetector
from emotion_model import EmotionModel
 
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "final_model.keras")
 
 
def annotate_image(image_bgr, detector, emotion_model):
    frame_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    boxes = detector.detect(frame_rgb)
 
    results = []
    for (x1, y1, x2, y2) in boxes:
        face_crop = frame_rgb[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue
 
        label, confidence, all_probs = emotion_model.predict(face_crop)
        results.append({"box": (x1, y1, x2, y2), "label": label, "confidence": confidence, "probs": all_probs})
 
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)
        text = f"{label} ({confidence:.0%})"
        cv2.putText(
            image_bgr, text, (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2,
        )
 
    return image_bgr, results
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", default=None, help="Path to save annotated image (default: <input>_annotated.jpg)")
    args = parser.parse_args()
 
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input image not found: {args.input}")
 
    output_path = args.output or os.path.splitext(args.input)[0] + "_annotated.jpg"
 
    detector = FaceDetector(min_confidence=0.6)
    emotion_model = EmotionModel(MODEL_PATH)
 
    image_bgr = cv2.imread(args.input)
    if image_bgr is None:
        raise ValueError(f"Could not read image (unsupported format or corrupt file): {args.input}")
 
    annotated, results = annotate_image(image_bgr, detector, emotion_model)
 
    if not results:
        print("No faces detected.")
    else:
        for i, r in enumerate(results):
            print(f"Face {i+1}: {r['label']} ({r['confidence']:.1%})")
 
    cv2.imwrite(output_path, annotated)
    print(f"Saved: {output_path}")
    detector.close()
 
 
if __name__ == "__main__":
    main()