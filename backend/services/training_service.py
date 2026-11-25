"""Single-job background training manager with persisted progress."""
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from datetime import datetime,timezone
from backend.config import REPORTS,MODELS
from backend.utils.helpers import read_json,write_json

_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='breed-training')
_lock=Lock()
_status={'status':'idle','model_type':None,'stage':'idle','current_epoch':0,'total_epochs':0,'accuracy':None,'validation_accuracy':None,'loss':None,'validation_loss':None,'error':None,'updated_at':None,'history':[]}

def _persist(): write_json(_status,REPORTS/'training_status.json')

def get_status():
    stored=read_json(REPORTS/'training_status.json',{}) or {}
    with _lock:
        if _status.get('status') != 'idle':
            return dict(_status)
        # A background worker cannot survive an API process restart. Don't
        # report a persisted running job forever when there is no worker.
        if stored.get('status') == 'running':
            stored.update({'status':'failed','stage':'interrupted','error':'The API restarted before this training job finished.'})
            write_json(stored,REPORTS/'training_status.json')
        return dict(stored or _status)

def start_training(model_type,epochs=12,batch_size=32,pretrained=True,augmentation_count=2,strategy='combined'):
    if model_type not in {'original','augmented'}: raise ValueError('model_type must be original or augmented.')
    if not 1<=epochs<=100: raise ValueError('epochs must be between 1 and 100.')
    if batch_size not in {8,16,32,64}: raise ValueError('batch_size must be 8, 16, 32, or 64.')
    with _lock:
        if _status['status']=='running': raise RuntimeError('A training job is already running.')
        _status.update({'status':'running','model_type':model_type,'stage':'queued','current_epoch':0,'total_epochs':epochs,'accuracy':None,'validation_accuracy':None,'loss':None,'validation_loss':None,'error':None,'updated_at':datetime.now(timezone.utc).isoformat(),'history':[]}); _persist()
    from backend.services.activity_service import record_activity
    record_activity('training',f'Started the {model_type} model training job.',{'model_type':model_type,'epochs':epochs})
    _executor.submit(_run,model_type,epochs,batch_size,pretrained,augmentation_count,strategy)
    return get_status()

def _update(values):
    with _lock:
        _status.update(values)
        if values.get('stage') == 'training' and values.get('current_epoch'):
            point={k:values.get(k) for k in ('current_epoch','accuracy','validation_accuracy','loss','validation_loss')}
            history=_status.setdefault('history',[])
            if not history or history[-1].get('current_epoch') != point['current_epoch']: history.append(point)
            else: history[-1]=point
        _status['updated_at']=datetime.now(timezone.utc).isoformat()
        _persist()

def _run(model_type,epochs,batch_size,pretrained,augmentation_count,strategy):
    try:
        from backend.ml.train import train_model
        result=train_model(model_type,epochs,batch_size,pretrained,_update,augmentation_count,strategy)
        _update({'status':'completed','stage':'completed','model_path':result['model_path'],'class_names':result['class_names'],'training_samples':result['training_samples'],'validation_samples':result['validation_samples']})
        from backend.services.activity_service import record_activity
        record_activity('training',f'Completed {model_type} model training.',{'model_type':model_type})
    except Exception as exc:
        _update({'status':'failed','stage':'failed','error':str(exc)})
        from backend.services.activity_service import record_activity
        record_activity('training',f'{model_type.title()} model training failed.',{'error':str(exc)})
