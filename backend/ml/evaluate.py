"""Evaluate saved models on the same untouched test split."""
from pathlib import Path
import gc
import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix,classification_report
from backend.config import MODELS,REPORTS,CONFUSION,GRAPHS
from backend.utils.helpers import read_json,write_json
from backend.ml.train import _paths_and_labels
from backend.services.dataset_service import create_splits

def _predict_paths(model,paths,batch_size):
    """Run bounded eager batches to avoid tf.data iterator aborts on macOS."""
    import tensorflow as tf
    outputs=[]
    for start in range(0,len(paths),batch_size):
        images=[]
        for path in paths[start:start+batch_size]:
            image=tf.io.decode_image(tf.io.read_file(path),channels=3,expand_animations=False)
            image=tf.image.resize(image,[224,224]); image=tf.cast(image,tf.float32)/255.0
            images.append(image.numpy())
        probabilities=model(np.stack(images),training=False).numpy()
        outputs.append(probabilities)
    return np.concatenate(outputs,axis=0)

def evaluate_model(name,batch_size=16):
    model_path=MODELS/f'{name}.keras'; config=read_json(MODELS/f'{name}_config.json',None)
    if not model_path.exists() or not config: return None
    import tensorflow as tf
    # Evaluate from the verified original test paths. The processed image
    # cache is only an optimization and can be incomplete after an interrupted
    # preprocessing run, including zero-byte files that TensorFlow cannot decode.
    classes=config['class_names']; splits=create_splits(); rows=splits['test']
    paths,y=_paths_and_labels(rows,classes)
    if not y: return None
    model=tf.keras.models.load_model(model_path)
    # Validation metrics describe the tuning split; test metrics below remain
    # the final, untouched comparison between models.
    val_paths,val_y=_paths_and_labels(splits['validation'],classes)
    if not val_y: raise ValueError(f'No usable validation images found for the {name} model.')
    val_probs=_predict_paths(model,val_paths,batch_size)
    val_loss=float(tf.keras.losses.sparse_categorical_crossentropy(np.asarray(val_y),val_probs).numpy().mean())
    probs=_predict_paths(model,paths,batch_size)
    pred=np.argmax(probs,axis=1); cm=confusion_matrix(y,pred,labels=list(range(len(classes))))
    precision,recall,f1,_=precision_recall_fscore_support(y,pred,average='weighted',zero_division=0)
    report={'model':name,'validation_accuracy':float(accuracy_score(val_y,np.argmax(val_probs,axis=1))),'validation_loss':val_loss,'validation_samples':len(val_y),'accuracy':float(accuracy_score(y,pred)),'precision':float(precision),'recall':float(recall),'f1_score':float(f1),'test_samples':len(y),'class_names':classes,'confusion_matrix':cm.tolist(),'classification_report':classification_report(y,pred,labels=list(range(len(classes))),target_names=classes,zero_division=0,output_dict=True)}
    write_json(report,REPORTS/f'{name}_evaluation.json'); _plot_confusion(cm,classes,name)
    # Evaluation runs two full models in one API process. Release TensorFlow's
    # cached graph/model state between them to avoid a second-run memory spike.
    del model, probs, pred, cm
    tf.keras.backend.clear_session()
    gc.collect()
    return report

def evaluate_all():
    results=[]; errors=[]
    for name in ('original','augmented'):
        try:
            result=evaluate_model(name)
            if result: results.append(result)
        except Exception as exc:
            errors.append({'model':name,'error':str(exc)})
            import tensorflow as tf
            tf.keras.backend.clear_session(); gc.collect()
    comparison=[]
    for r in results: comparison.append({k:r[k] for k in ('model','validation_accuracy','validation_loss','validation_samples','accuracy','precision','recall','f1_score','test_samples')})
    result={'pending':len(results)<2,'models':results,'comparison':comparison,'errors':errors}
    if len(results)==2:
        import pandas as pd
        pd.DataFrame(comparison).to_csv(REPORTS/'model_comparison.csv',index=False)
        _plot_comparison(comparison)
    write_json(result,REPORTS/'evaluation_results.json')
    return result

def _plot_confusion(cm,labels,name):
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt,seaborn as sns
    CONFUSION.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(max(10,len(labels)*.45),max(8,len(labels)*.4)))
    sns.heatmap(cm,annot=len(labels)<=15,fmt='d',cmap='Blues',xticklabels=labels,yticklabels=labels,ax=ax)
    ax.set(xlabel='Predicted breed',ylabel='True breed',title=f'{name.title()} model confusion matrix'); fig.tight_layout(); fig.savefig(CONFUSION/f'{name}_confusion_matrix.png',dpi=140); plt.close(fig)

def _plot_comparison(rows):
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    GRAPHS.mkdir(parents=True,exist_ok=True); keys=('accuracy','precision','recall','f1_score')
    fig,ax=plt.subplots(figsize=(8,4)); x=np.arange(len(keys)); width=.35
    for i,row in enumerate(rows): ax.bar(x+(i-.5)*width,[row[k] for k in keys],width,label=row['model'].title())
    ax.set_xticks(x,keys); ax.set_ylim(0,1); ax.set_ylabel('Score'); ax.legend(); fig.tight_layout(); fig.savefig(GRAPHS/'model_comparison.png',dpi=140); plt.close(fig)
