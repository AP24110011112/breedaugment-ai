"""Reusable preprocessing operations."""
from PIL import Image
from io import BytesIO
import numpy as np
from backend.config import MODEL_INPUT_SIZE

def preprocess_bytes(image_bytes: bytes) -> np.ndarray:
    with Image.open(BytesIO(image_bytes)) as image:
        image=image.convert('RGB').resize(MODEL_INPUT_SIZE,Image.Resampling.LANCZOS)
        return np.asarray(image,dtype=np.float32)/255.0
