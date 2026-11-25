"""Shared training code for the paired model experiment."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from config import MODELS,REPORTS,GRAPHS
from utils.io import list_images,write_json
from src.model import build_model

def load_images(folder,class_names):
    x=[]; y=[]
    for idx,name in enumerate(class_names):
        for p in list_images(Path(folder)/name):
            try:
                with Image.open(p) as image: x.append(np.asarray(image.convert('RGB').resize((224,224)),dtype=np.float32)/255.); y.append(idx)
            except OSError: continue
    return np.asarray(x,dtype=np.float32),np.asarray(y,dtype=np.int64)

def train(train_dir,val_dir,class_names,name,epochs=12,batch_size=32,pretrained=True):
    x,y=load_images(train_dir,class_names); xv,yv=load_images(val_dir,class_names)
    if len(x)<2 or len(xv)<1: raise ValueError('Not enough train/validation images. Add real photos per class.')
    model=build_model(len(class_names),pretrained)
    path=MODELS/f'{name}.keras'; callbacks=[
      __import__('tensorflow').keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True),
      __import__('tensorflow').keras.callbacks.ModelCheckpoint(path,monitor='val_loss',save_best_only=True)]
    hist=model.fit(x,y,validation_data=(xv,yv),epochs=epochs,batch_size=batch_size,callbacks=callbacks,shuffle=True,verbose=1)
    model.save(path); history={k:[float(v) for v in vals] for k,vals in hist.history.items()}
    write_json({'class_names':class_names,'image_size':[224,224],'architecture':'MobileNetV2','pretrained':pretrained,'model_name':name},MODELS/f'{name}_config.json')
    write_json(history,REPORTS/f'{name}_history.json'); plot_history(history,name)
    return history

def plot_history(history,name):
    import matplotlib.pyplot as plt
    GRAPHS.mkdir(parents=True,exist_ok=True)
    for metric in ('accuracy','loss'):
        fig,ax=plt.subplots(figsize=(7,4))
        if metric in history: ax.plot(history[metric],label='train')
        if 'val_'+metric in history: ax.plot(history['val_'+metric],label='validation')
        ax.set(xlabel='Epoch',ylabel=metric.title()); ax.legend(); fig.tight_layout(); fig.savefig(GRAPHS/f'{name}_{metric}.png',dpi=140); plt.close(fig)
