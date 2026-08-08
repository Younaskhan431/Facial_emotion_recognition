Facial Emotion Recognition

A real-time facial emotion recognition system that detects faces and classifies emotions from images, videos, and live webcam feeds. Built with a fine-tuned ResNet50 classifier trained on RAF-DB, paired with MediaPipe for face detection.

Overview

This project combines a deep learning emotion classifier with a lightweight, production-style inference pipeline. It supports three input modes — single images, video files, and real-time webcam — through both command-line scripts and (optionally) a Streamlit web interface.

Detected emotions: Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral

Features
Face detection via MediaPipe (fast, CPU-friendly, works across varying face distances/angles)
Emotion classification via a fine-tuned ResNet50 model trained on RAF-DB (7 classes)
Three inference modes:
Single image → annotated output image with per-face predictions
Video file → fully annotated output video, with optional frame extraction
Live webcam → real-time detection and classification overlay
Configurable performance/quality trade-off for video via frame-skipping
Cross-platform video codec handling (H.264 with automatic fallback)
Demo
Input	Output
Image	Bounding box + emotion label + confidence per detected face
Video	Annotated video with per-frame predictions
Webcam	Live overlay, updated continuously
Model Details
	
Architecture	ResNet50 (transfer learning, fine-tuned)
Dataset	RAF-DB
Classes	7 (Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral)
Input size	224 × 224 RGB
Training accuracy	~95% (RAF-DB test set)

Note on real-world performance: the model was trained on RAF-DB, which consists of pre-cropped, curated face images. Performance on live webcam or in-the-wild photos/videos can be noticeably lower than the reported training accuracy, due to differences in framing, lighting, and image quality between the training distribution and real-world capture conditions. This is a known and expected form of domain gap, not a bug in the inference pipeline.

Tech Stack
Deep Learning: TensorFlow / Keras
Face Detection: MediaPipe
Computer Vision: OpenCV
Web Interface (optional): Streamlit
Language: Python 3.10+
Project Structure
facial-emotion-recognition/
├── models/
│   └── final_model.keras        # Trained emotion classifier
├── src/
│   ├── emotion_model.py         # Model loading + prediction
│   ├── face_detector.py         # MediaPipe face detection wrapper
│   ├── image_inference.py       # CLI: single-image inference
│   ├── video_inference.py       # CLI: video file inference
│   └── realtime.py              # Live webcam inference
├── requirements.txt
└── README.md
Installation
bash
git clone https://github.com/<your-username>/facial-emotion-recognition.git
cd facial-emotion-recognition

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt

The trained model (models/final_model.keras) is tracked with Git LFS — see Model File below if it doesn't download automatically.

Usage
Image inference
bash
python src/image_inference.py --input path/to/photo.jpg --output path/to/result.jpg

Detects all faces in the image, prints predicted emotion + confidence for each, and saves an annotated copy.

Video inference
bash
python src/video_inference.py --input path/to/clip.mp4 --output path/to/result.mp4

Processes every frame and writes an annotated copy.

Options:

--skip-frames N — only run detection/classification on every (N+1)th frame, reusing prior results in between, for faster processing on CPU
--save-frames — also save individual annotated frames as .jpg files
--frame-interval N — used with --save-frames, saves every Nth frame instead of all of them
Real-time webcam
bash
python src/realtime.py

Opens a live window with detection and classification running on your webcam feed. Press q to quit.

Model File

final_model.keras is tracked with Git LFS due to its size. After cloning:

bash
git lfs install
git lfs pull

If you're setting this up from your own training run rather than cloning, place your trained model at models/final_model.keras before running any script.

Known Limitations
Domain gap: trained on curated dataset images; real-world webcam/photo accuracy is lower than reported training accuracy (see Model Details)
Prediction stability: frame-by-frame inference with no temporal smoothing can cause the predicted label to shift between visually similar emotions (e.g. Neutral/Sadness) even when the subject's expression is constant
False positives: MediaPipe's face detector can occasionally misidentify background elements as faces, particularly in cluttered scenes
Future Improvements
Temporal smoothing across frames for more stable live predictions
Fine-tuning threshold/confidence calibration for real-world conditions
Deployment as a hosted web demo
License

This project is provided for educational and portfolio purposes.
