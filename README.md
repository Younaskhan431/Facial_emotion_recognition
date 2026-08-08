# Facial Emotion Recognition

A real-time facial emotion recognition system that detects faces and classifies emotions from images, videos, and live webcam feeds. Built with a fine-tuned ResNet50 classifier trained on RAF-DB, paired with MediaPipe for face detection.

## Overview

This project combines a deep learning emotion classifier with a lightweight, production-style inference pipeline. It supports three input modes — single images, video files, and real-time webcam — through command-line scripts.

**Detected emotions:** Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral

## Features

- **Face detection** via MediaPipe (fast, CPU-friendly, works across varying face distances/angles)
- **Emotion classification** via a fine-tuned ResNet50 model trained on RAF-DB (7 classes)
- **Three inference modes:**
  - Single image → annotated output image with per-face predictions
  - Video file → fully annotated output video, with optional frame extraction
  - Live webcam → real-time detection and classification overlay
- **Configurable performance/quality trade-off** for video via frame-skipping
- **Cross-platform video codec handling** (H.264 with automatic fallback)

## Demo

| Input | Output |
|---|---|
| Image | Bounding box + emotion label + confidence per detected face |
| Video | Annotated video with per-frame predictions |
| Webcam | Live overlay, updated continuously |

## Model Details

| | |
|---|---|
| Architecture | ResNet50 (transfer learning, fine-tuned) |
| Dataset | RAF-DB |
| Classes | 7 (Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral) |
| Input size | 224 × 224 RGB |
| Training accuracy | ~95% (RAF-DB test set) |

**Note on real-world performance:** the model was trained on RAF-DB, which consists of pre-cropped, curated face images. Performance on live webcam or in-the-wild photos/videos can be noticeably lower than the reported training accuracy, due to differences in framing, lighting, and image quality between the training distribution and real-world capture conditions. This is a known and expected form of domain gap, not a bug in the inference pipeline.

## Tech Stack

- **Deep Learning:** TensorFlow / Keras
- **Face Detection:** MediaPipe
- **Computer Vision:** OpenCV
- **Language:** Python 3.10+

## Project Structure
