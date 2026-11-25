"""Model evaluation endpoints."""
from fastapi import APIRouter,HTTPException
from backend.services.evaluation_service import get_results,run_evaluation

router=APIRouter(prefix='/api/evaluation',tags=['evaluation'])

@router.get('/results')
def results(): return get_results()

@router.post('/run')
def run():
    try: return run_evaluation()
    except Exception as exc: raise HTTPException(400,str(exc)) from exc
