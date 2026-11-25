"""Same-test-set evaluation and reports."""
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix,classification_report
from config import MODELS,REPORTS,CM,GRAPHS
from utils.io import write_json
from src.train import load_images

def evaluate(model_path,test_dir,class_names,name):
    if not Path(model_path).exists(): raise FileNotFoundError(f'Model is not trained yet: {model_path}')
    import tensorflow as tf
    x,y=load_images(test_dir,class_names)
    if not len(y): raise ValueError('Test split is empty.')
    model=tf.keras.models.load_model(model_path); pred=np.argmax(model.predict(x,verbose=0),axis=1)
    p,r,f,_=precision_recall_fscore_support(y,pred,average='weighted',zero_division=0)
    cm=confusion_matrix(y,pred,labels=list(range(len(class_names))))
    report={'model':name,'accuracy':float(accuracy_score(y,pred)),'precision_weighted':float(p),'recall_weighted':float(r),'f1_weighted':float(f),'test_samples':int(len(y)),'class_names':class_names,'confusion_matrix':cm.tolist(),'classification_report':classification_report(y,pred,labels=list(range(len(class_names))),target_names=class_names,zero_division=0,output_dict=True)}
    REPORTS.mkdir(parents=True,exist_ok=True); write_json(report,REPORTS/f'{name}_evaluation.json'); plot_cm(cm,class_names,name)
    return report

def plot_cm(matrix,labels,name):
    import matplotlib.pyplot as plt,seaborn as sns
    CM.mkdir(parents=True,exist_ok=True); fig,ax=plt.subplots(figsize=(max(6,len(labels)*.6),5))
    sns.heatmap(matrix,annot=len(labels)<=15,fmt='d',cmap='Blues',xticklabels=labels,yticklabels=labels,ax=ax)
    ax.set(xlabel='Predicted breed',ylabel='Actual breed',title=name); fig.tight_layout(); fig.savefig(CM/f'{name}_confusion_matrix.png',dpi=140); plt.close(fig)

def compare(original,augmented):
    import pandas as pd,matplotlib.pyplot as plt
    rows=[{k:r[k] for k in ('model','accuracy','precision_weighted','recall_weighted','f1_weighted')} for r in (original,augmented)]
    frame=pd.DataFrame(rows); frame.to_csv(REPORTS/'model_comparison.csv',index=False)
    GRAPHS.mkdir(parents=True,exist_ok=True); ax=frame.set_index('model')[['accuracy','precision_weighted','recall_weighted','f1_weighted']].plot(kind='bar',ylim=(0,1),figsize=(8,4)); ax.set_ylabel('Score'); ax.figure.tight_layout(); ax.figure.savefig(GRAPHS/'model_comparison.png',dpi=140); plt.close(ax.figure)
    return frame
