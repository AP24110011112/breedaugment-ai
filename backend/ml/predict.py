"""Prediction from saved Keras model bundle."""
from pathlib import Path
from io import BytesIO
import numpy as np
from PIL import Image
from backend.config import MODELS
from backend.utils.helpers import read_json


def predict_bytes(image_bytes,model_type='augmented'):
    model_path=MODELS/f'{model_type}.keras'; config=read_json(MODELS/f'{model_type}_config.json',None)
    if not model_path.exists() or not config: raise FileNotFoundError(f'No trained {model_type} model is available yet.')
    import tensorflow as tf
    with Image.open(BytesIO(image_bytes)) as image:
        image=image.convert('RGB').resize((224,224),Image.Resampling.LANCZOS)
        arr=np.asarray(image,dtype=np.float32)[None,...]/255.
    model=tf.keras.models.load_model(model_path); probs=model.predict(arr,verbose=0)[0]
    ids=np.argsort(probs)[::-1][:min(3,len(config['class_names']))]
    predictions=[{'breed':config['class_names'][int(i)],'confidence':float(probs[i])} for i in ids]
    return {'model':model_type,'predicted_breed':predictions[0]['breed'],'confidence':predictions[0]['confidence'],'top_predictions':predictions,'note':'Predictions are limited to breeds present in the trained dataset.'}
