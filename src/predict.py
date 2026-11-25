"""Load saved classifier and predict a pet photo."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from config import MODELS

def load_bundle(model_path):
    model_path=Path(model_path); cfg=model_path.with_name(model_path.stem+'_config.json')
    if not model_path.exists() or not cfg.exists(): raise FileNotFoundError('Train a model and save its configuration first.')
    import tensorflow as tf
    return tf.keras.models.load_model(model_path),json.loads(cfg.read_text())

def predict(model,image,class_names):
    if not isinstance(image,Image.Image): image=Image.open(image)
    arr=np.asarray(image.convert('RGB').resize((224,224)),dtype=np.float32)/255.
    scores=model.predict(arr[None,...],verbose=0)[0]; ids=np.argsort(scores)[::-1][:min(3,len(class_names))]
    return [{'breed':class_names[i],'confidence':float(scores[i])} for i in ids]
