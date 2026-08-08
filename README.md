## Installation

```bash
git clone https://github.com/Younaskhan431/Facial_emotion_recognition.git
cd Facial_emotion_recognition

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

### Image inference

```bash
python src/image_inference.py --input path/to/photo.jpg --output path/to/result.jpg
```

Detects all faces in the image, prints predicted emotion + confidence for each, and saves an annotated copy.

### Video inference

```bash
python src/video_inference.py --input path/to/clip.mp4 --output path/to/result.mp4
```

Processes every frame and writes an annotated copy.

**Options:**
- `--skip-frames N` — only run detection/classification on every (N+1)th frame, reusing prior results in between, for faster processing on CPU
- `--save-frames` — also save individual annotated frames as `.jpg` files
- `--frame-interval N` — used with `--save-frames`, saves every Nth frame instead of all of them

### Real-time webcam

```bash
python src/realtime.py
```

Opens a live window with detection and classification running on your webcam feed. Press `q` to quit.

## Known Limitations

- **Domain gap:** trained on curated dataset images; real-world webcam/photo accuracy is lower than reported training accuracy (see Model Details above)
- **Prediction stability:** frame-by-frame inference with no temporal smoothing can cause the predicted label to shift between visually similar emotions (e.g. Neutral/Sadness) even when the subject's expression is constant
- **False positives:** MediaPipe's face detector can occasionally misidentify background elements as faces, particularly in cluttered scenes

## Future Improvements

- Temporal smoothing across frames for more stable live predictions
- Fine-tuning threshold/confidence calibration for real-world conditions
- Deployment as a hosted web demo

## License

This project is provided for educational and portfolio purposes.
