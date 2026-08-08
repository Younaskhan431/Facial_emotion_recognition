"""
Loads the trained emotion classifier and runs prediction on a single face crop.

Matches training exactly:
- Input size: 224x224, RGB
- Class order: index-aligned with the RAF-DB labels used in training (0-6)
- No manual normalization here — the model's Lambda layer already bakes in
  the backbone-specific preprocess_input (see notebook cell 29). Feeding
  already-normalized pixels in again would silently break predictions.
"""

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import resnet50

IMG_SIZE = 224

CLASS_NAMES = [
    "Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral",
]


class EmotionModel:
    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={"preprocess_input": resnet50.preprocess_input},
        )

    def preprocess(self, face_img: np.ndarray) -> np.ndarray:
        """face_img: RGB numpy array (any size, any dtype 0-255). Returns a
        (1, 224, 224, 3) float32 batch ready to feed straight into the model."""
        img = Image.fromarray(face_img).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.float32)
        return np.expand_dims(arr, axis=0)

    def predict(self, face_img: np.ndarray):
        """Returns (label: str, confidence: float, all_probs: dict)."""
        batch = self.preprocess(face_img)
        probs = self.model.predict(batch, verbose=0)[0]
        idx = int(np.argmax(probs))
        all_probs = {name: float(p) for name, p in zip(CLASS_NAMES, probs)}
        return CLASS_NAMES[idx], float(probs[idx]), all_probs
