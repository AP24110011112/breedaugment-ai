"""Background model training endpoints."""
from fastapi import APIRouter,HTTPException
from pydantic import BaseModel,Field
from backend.services.training_service import start_training,get_status

router=APIRouter(prefix='/api/training',tags=['training'])

class TrainingRequest(BaseModel):
    model_type:str=Field(pattern='^(original|augmented)$')
    epochs:int=Field(default=12,ge=1,le=100)
    batch_size:int=Field(default=32,ge=8,le=64)
    pretrained:bool=True
    augmentation_count:int=Field(default=2,ge=0,le=20)
    strategy:str='combined'

@router.post('/start')
def start(request:TrainingRequest):
    try:
        if request.strategy not in {'lighting','rotation','perspective','scaling','blur_noise','occlusion','combined'}: raise ValueError('Unknown augmentation strategy.')
        return start_training(request.model_type,request.epochs,request.batch_size,request.pretrained,request.augmentation_count,request.strategy)
    except RuntimeError as exc: raise HTTPException(409,str(exc)) from exc
    except ValueError as exc: raise HTTPException(400,str(exc)) from exc

@router.get('/status')
def status(): return get_status()
