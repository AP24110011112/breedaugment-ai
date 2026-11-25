"""Image augmentation endpoints."""
from fastapi import APIRouter,UploadFile,File,Form,HTTPException
from pydantic import BaseModel,Field
from backend.services.augmentation_service import generate_preview,list_samples,PIPELINES,add_previews_to_dataset
from backend.utils.helpers import validate_image_bytes

router=APIRouter(prefix='/api/augmentation',tags=['augmentation'])

class PreviewSelection(BaseModel):
    filename:str=Field(min_length=1,max_length=255)
    strategy:str

class AddToDatasetRequest(BaseModel):
    breed:str=Field(min_length=1,max_length=64)
    original_filename:str=Field(default='pet-image.jpg',max_length=255)
    images:list[PreviewSelection]=Field(min_length=1,max_length=12)
    create_category:bool=False

@router.post('/generate')
async def generate(image:UploadFile=File(...),strategy:str=Form('combined'),count:int=Form(3)):
    if strategy not in PIPELINES: raise HTTPException(400,f'Strategy must be one of: {", ".join(PIPELINES)}')
    try:
        data=await image.read(); validate_image_bytes(data,image.filename or 'upload.jpg')
        generated=generate_preview(data,strategy,count)
        from backend.services.activity_service import record_activity
        record_activity('augmentation',f'Generated {len(generated)} {strategy} preview(s).',{'strategy':strategy,'count':len(generated)})
        return {'original_filename':image.filename,'strategy':strategy,'count':len(generated),'images':generated}
    except ValueError as exc: raise HTTPException(400,str(exc)) from exc

@router.get('/samples')
def samples(): return {'samples':list_samples()}

@router.post('/add-to-dataset')
def add_to_dataset(request:AddToDatasetRequest):
    try:
        return add_previews_to_dataset(request.breed,request.original_filename,request.images,request.create_category)
    except FileNotFoundError as exc: raise HTTPException(404,str(exc)) from exc
    except ValueError as exc: raise HTTPException(400,str(exc)) from exc
