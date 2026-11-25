"""Memory-efficient tf.data training loop with progress callbacks."""
from pathlib import Path
from collections import Counter
import numpy as np
from backend.config import PROCESSED,MODELS,REPORTS,GRAPHS
import tensorflow as tf
from backend.ml.model import build_model
from backend.utils.helpers import read_json,write_json

def _paths_and_labels(rows,class_names):
    mapping={name:i for i,name in enumerate(class_names)}
    paths=[]; labels=[]
    for row in rows:
        p=Path(row['path'])
        if p.is_file() and row['label'] in mapping:
            paths.append(str(p)); labels.append(mapping[row['label']])
    return paths,labels

def _dataset(paths,labels,batch_size,shuffle=False):
    ds=tf.data.Dataset.from_tensor_slices((paths,np.asarray(labels,dtype=np.int32)))
    def load(path,label):
        data=tf.io.read_file(path); image=tf.io.decode_image(data,channels=3,expand_animations=False)
        image.set_shape([None,None,3]); image=tf.image.resize(image,[224,224]); image=tf.cast(image,tf.float32)/255.0
        return image,label
    ds=ds.map(load,num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle: ds=ds.shuffle(min(len(paths),10000),reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def _directory_records(directory):
    out=[]
    for breed in sorted(p for p in directory.iterdir() if p.is_dir()):
        out.extend({'path':str(p),'label':breed.name} for p in sorted(breed.glob('*')) if p.suffix.lower() in {'.jpg','.jpeg','.png'})
    return out

def train_model(model_type,epochs,batch_size,pretrained,progress_callback,augmented_count=2,strategy='combined'):
    from backend.services.dataset_service import create_splits
    from backend.services.augmentation_service import training_variants
    splits=create_splits()
    class_names=sorted({r['label'] for r in splits['train']})
    train_rows=list(splits['train']);
    if model_type=='augmented':
        progress_callback({'stage':'preparing augmentation','current_epoch':0,'total_epochs':epochs})
        variants=training_variants(augmented_count,strategy)
        train_sources={str(Path(r['path']).resolve()) for r in splits['train']}
        for row in variants:
            if str(Path(row['source']).resolve()) in train_sources and row['label'] in class_names:
                train_rows.append({'path':row['path'],'label':row['label']})
    train_paths,train_y=_paths_and_labels(train_rows,class_names)
    val_paths,val_y=_paths_and_labels(splits['validation'],class_names)
    if len(train_y)<2 or not len(val_y): raise ValueError('Insufficient train or validation images.')
    train_ds=_dataset(train_paths,train_y,batch_size,True); val_ds=_dataset(val_paths,val_y,batch_size)
    model=build_model(len(class_names),pretrained)
    MODELS.mkdir(parents=True,exist_ok=True); REPORTS.mkdir(parents=True,exist_ok=True)
    name='original' if model_type=='original' else 'augmented'; model_path=MODELS/f'{name}.keras'
    class Progress(tf.keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs=None):
            logs=logs or {}
            progress_callback({'stage':'training','current_epoch':epoch+1,'total_epochs':epochs,'accuracy':float(logs.get('accuracy',0)),'validation_accuracy':float(logs.get('val_accuracy',0)),'loss':float(logs.get('loss',0)),'validation_loss':float(logs.get('val_loss',0))})
    callbacks=[Progress(),tf.keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss',save_best_only=True),tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=4,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.3,patience=2,min_lr=1e-6)]
    history=model.fit(train_ds,validation_data=val_ds,epochs=epochs,callbacks=callbacks,verbose=0)
    model.save(model_path)
    config={'class_names':class_names,'image_size':[224,224],'architecture':'MobileNetV2','pretrained':pretrained,'model_type':model_type,'training_samples':len(train_paths),'validation_samples':len(val_paths),'augmentation_count':augmented_count if model_type=='augmented' else 0}
    write_json(config,MODELS/f'{name}_config.json')
    hist={k:[float(x) for x in vals] for k,vals in history.history.items()}; write_json(hist,REPORTS/f'{name}_history.json')
    _plot_history(hist,name)
    return {'model_type':model_type,'class_names':class_names,'history':hist,'model_path':str(model_path),'training_samples':len(train_paths),'validation_samples':len(val_paths)}

def _plot_history(history,name):
    import matplotlib
    # Training runs in a worker thread and the API is headless. A GUI backend
    # can block the worker after the final epoch and leave the API unresponsive.
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    GRAPHS.mkdir(parents=True,exist_ok=True)
    for metric in ('accuracy','loss'):
        fig,ax=plt.subplots(figsize=(7,4))
        if metric in history: ax.plot(history[metric],label='Train')
        if 'val_'+metric in history: ax.plot(history['val_'+metric],label='Validation')
        ax.set(xlabel='Epoch',ylabel=metric.title(),title=f'{name.title()} model {metric}'); ax.legend(); fig.tight_layout(); fig.savefig(GRAPHS/f'{name}_{metric}.png',dpi=140); plt.close(fig)
