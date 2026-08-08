"""
Run face detection + emotion recognition on a video file, frame by frame,
and save an annotated copy.
 
Usage:
    python src/video_inference.py --input path/to/clip.mp4 --output path/to/result.mp4
 
Note: this processes every frame through the model, so it will take noticeably
longer than the video's own runtime on CPU. Use --skip-frames to sample less
often if you just want a quick preview.
"""
 
import argparse
import os
import sys
import time
 
import cv2
 
sys.path.append(os.path.dirname(__file__))
from face_detector import FaceDetector
from emotion_model import EmotionModel
from image_inference import annotate_image
 
 
def draw_annotations_from_results(frame, results):
    annotated = frame.copy()
    for result in results:
        x1, y1, x2, y2 = result["box"]
        label = result["label"]
        confidence = result["confidence"]
 
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
        text = f"{label} ({confidence:.0%})"
        cv2.putText(
            annotated,
            text,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 0),
            2,
        )
    return annotated
 
 
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "final_model.keras")
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", default=None, help="Path to save annotated video (default: <input>_annotated.mp4)")
    parser.add_argument("--skip-frames", type=int, default=0,
                         help="Reuse the previous frame's boxes/labels for this many frames "
                              "between fresh predictions, to speed up processing (0 = predict every frame)")
    parser.add_argument("--save-frames", action="store_true",
                         help="Also save each processed frame as a separate .jpg file, in a "
                              "'<output_name>_frames' folder next to the output video.")
    parser.add_argument("--frame-interval", type=int, default=1,
                         help="When --save-frames is used, only save every Nth processed frame "
                              "(default: every frame). E.g. 30 saves roughly one frame per second at 30fps video.")
    args = parser.parse_args()
 
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input video not found: {args.input}")
 
    output_path = args.output or os.path.splitext(args.input)[0] + "_annotated.mp4"
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
 
    frames_dir = None
    if args.save_frames:
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        frames_dir = os.path.join(output_dir or ".", f"{base_name}_frames")
        os.makedirs(frames_dir, exist_ok=True)
 
    detector = FaceDetector(min_confidence=0.6)
    emotion_model = EmotionModel(MODEL_PATH)
 
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {args.input}")
 
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 
    # avc1 (H.264) plays natively in Windows' default video player, unlike
    # mp4v which often fails to decode outside VLC/ffmpeg-based players.
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        # Some OpenCV builds don't ship an H.264 encoder — fall back to mp4v
        # so the script still works, just less compatible with Windows' player.
        print("Warning: avc1 encoder unavailable, falling back to mp4v (use VLC to play the result).")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to initialize video writer for: {output_path}")
 
    frame_idx = 0
    saved_frame_count = 0
    previous_results = []
    start = time.time()
 
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
 
            if args.skip_frames > 0 and frame_idx % (args.skip_frames + 1) != 0:
                # Reuse the prior annotations for skipped frames to keep the output consistent.
                if frame_idx == 0:
                    annotated, previous_results = annotate_image(frame, detector, emotion_model)
                else:
                    annotated = draw_annotations_from_results(frame, previous_results)
            else:
                annotated, previous_results = annotate_image(frame, detector, emotion_model)
 
            writer.write(annotated)
 
            if frames_dir and frame_idx % args.frame_interval == 0:
                frame_path = os.path.join(frames_dir, f"frame_{frame_idx:05d}.jpg")
                cv2.imwrite(frame_path, annotated)
                saved_frame_count += 1
 
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames...")
    finally:
        cap.release()
        writer.release()
        detector.close()
 
    elapsed = time.time() - start
    print(f"Done. Saved: {output_path} ({frame_idx} frames in {elapsed:.1f}s)")
    if frames_dir:
        print(f"Saved {saved_frame_count} individual frame(s) to: {frames_dir}")
 
 
if __name__ == "__main__":
    main()