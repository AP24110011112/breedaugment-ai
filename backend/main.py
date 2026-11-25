"""FastAPI application. Supports both `uvicorn main:app` from backend/ and backend.main:app from the repo root."""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.config import ORIGINAL,AUGMENTED,OUTPUTS,SAMPLES,GRAPHS,REPORTS,CONFUSION,HOST_ARCH,IS_APPLE_SILICON,IS_MAC_INTEL,MODELS
from backend.api import dataset_routes,augmentation_routes,training_routes,evaluation_routes,prediction_routes
from backend.services.dataset_service import dataset_stats
from backend.services.evaluation_service import get_results
from backend.services.activity_service import recent_activities
from backend.utils.helpers import read_json

app=FastAPI(title='BreedAugment AI API',version='1.0.0',description='Local-first dataset, augmentation, TensorFlow training, evaluation, and breed prediction API.')
_default_origins='http://localhost:5173,http://127.0.0.1:5173'
_frontend_origins=[origin.strip().rstrip('/') for origin in os.getenv('FRONTEND_ORIGINS',_default_origins).split(',') if origin.strip()]
app.add_middleware(CORSMiddleware,allow_origins=_frontend_origins,allow_credentials=True,allow_methods=['*'],allow_headers=['*'])
app.include_router(dataset_routes.router); app.include_router(augmentation_routes.router); app.include_router(training_routes.router); app.include_router(evaluation_routes.router); app.include_router(prediction_routes.router)
for url,directory in (('/files/dataset/original',ORIGINAL),('/files/dataset/augmented',AUGMENTED),('/files/outputs/samples',SAMPLES),('/files/outputs/augmentation_samples',OUTPUTS/'augmentation_samples'),('/files/outputs/graphs',GRAPHS),('/files/outputs/reports',REPORTS),('/files/outputs/confusion_matrix',CONFUSION)):
    directory.mkdir(parents=True,exist_ok=True); app.mount(url,StaticFiles(directory=directory),name='files_'+url.replace('/','_'))

@app.get('/api/health')
def health():
    return {'status':'ok','service':'BreedAugment AI','architecture':HOST_ARCH,'apple_silicon':IS_APPLE_SILICON,'intel_mac':IS_MAC_INTEL}

@app.get('/api/dashboard/stats')
def dashboard_stats():
    stats=dataset_stats(); report=get_results(); models={}
    for name in ('original','augmented'):
        cfg=read_json(MODELS/f'{name}_config.json',{}) or {}
        evaluation=next((x for x in report.get('models',[]) if x.get('model')==name),None)
        models[name]={'trained':(MODELS/f'{name}.keras').exists(),'accuracy':evaluation.get('accuracy') if evaluation else None,'precision':evaluation.get('precision') if evaluation else None,'recall':evaluation.get('recall') if evaluation else None,'f1_score':evaluation.get('f1_score') if evaluation else None,'config':cfg}
    latest=read_json(REPORTS/'latest_prediction.json',None)
    return {'original_images':stats['images'],'augmented_images':sum(1 for p in AUGMENTED.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'}),'breed_classes':stats['classes'],'breeds':stats['breeds'],'splits':stats.get('splits',{}),'training_status':read_json(REPORTS/'training_status.json',{'status':'idle'}),'models':models,'latest_prediction':latest,'evaluation_pending':report.get('pending',True),'recent_activities':recent_activities()}

@app.get('/api/models/status')
def model_status():
    result={}
    for name in ('original','augmented'):
        model=MODELS/f'{name}.keras'; cfg=read_json(MODELS/f'{name}_config.json',None)
        result[name]={'available':model.exists() and cfg is not None,'model_path':str(model) if model.exists() else None,'class_names':cfg.get('class_names',[]) if cfg else []}
    return result
