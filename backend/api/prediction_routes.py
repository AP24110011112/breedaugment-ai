"""Breed prediction endpoint."""
from fastapi import APIRouter,UploadFile,File,Form,HTTPException
from backend.services.prediction_service import predict_bytes
from backend.utils.helpers import validate_image_bytes,write_json
from backend.config import REPORTS

router=APIRouter(prefix='/api/prediction',tags=['prediction'])

@router.post('/predict')
async def predict(image:UploadFile=File(...),model_type:str=Form('augmented')):
    if model_type not in {'original','augmented'}: raise HTTPException(400,'Model must be original or augmented.')
    try:
        data=await image.read(); validate_image_bytes(data,image.filename or 'upload.jpg')
        result=predict_bytes(data,model_type); write_json(result,REPORTS/'latest_prediction.json')
        from backend.services.activity_service import record_activity
        record_activity('prediction',f'Predicted {result["predicted_breed"]}.',{'breed':result['predicted_breed'],'confidence':result['confidence']})
        return result
    except FileNotFoundError as exc: raise HTTPException(404,str(exc)) from exc
    except ValueError as exc: raise HTTPException(400,str(exc)) from exc
    except Exception as exc: raise HTTPException(500,f'Prediction failed: {exc}') from exc
