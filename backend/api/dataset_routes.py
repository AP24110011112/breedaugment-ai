"""Dataset endpoints."""
from fastapi import APIRouter,UploadFile,File,Form,HTTPException
from pydantic import BaseModel,Field
from backend.services.dataset_service import get_breeds,dataset_stats,upload_images,delete_image,validate_dataset,create_splits,create_breed_category,DatasetError
from backend.config import ORIGINAL
from backend.utils.helpers import safe_breed_name,ALLOWED_EXTENSIONS

router=APIRouter(prefix='/api/dataset',tags=['dataset'])

class BreedCategoryRequest(BaseModel):
    breed:str=Field(min_length=1,max_length=64)

@router.post('/breeds')
def add_breed(request:BreedCategoryRequest):
    try:
        result=create_breed_category(request.breed)
        from backend.services.activity_service import record_activity
        if result['created']: record_activity('dataset',f'Created breed category {result["breed"]}.')
        return result
    except ValueError as exc: raise HTTPException(400,str(exc)) from exc

@router.get('/breeds')
def breeds(): return {'breeds':get_breeds()}

@router.get('/stats')
def stats(): return dataset_stats()

@router.post('/upload')
async def upload(breed:str=Form(...),files:list[UploadFile]=File(...)):
    if not files: raise HTTPException(400,'Select at least one image.')
    if len(files)>50: raise HTTPException(400,'Upload at most 50 images per request.')
    payload=[]; total_bytes=0
    for file in files:
        try:
            if file.filename is None or file.content_type not in {'image/jpeg','image/png','image/jpg','application/octet-stream'}: raise ValueError('Upload JPG, JPEG, or PNG images only.')
            data=await file.read(); total_bytes+=len(data)
            if total_bytes>100*1024*1024: raise ValueError('Total upload size cannot exceed 100 MB per request.')
            payload.append((file.filename,data))
        except ValueError as exc: raise HTTPException(400,str(exc)) from exc
    try:
        saved=upload_images(breed,payload)
        # New originals invalidate the current split; a new one is created before training.
        from backend.config import SPLITS
        (SPLITS/'manifest.json').unlink(missing_ok=True)
        from backend.services.activity_service import record_activity
        record_activity('upload',f'Uploaded {len(saved)} photo(s) to {safe_breed_name(breed)}.',{'count':len(saved),'breed':safe_breed_name(breed)})
        return {'uploaded':len(saved),'images':saved,'breed':safe_breed_name(breed)}
    except ValueError as exc: raise HTTPException(400,str(exc)) from exc

@router.get('/images/{breed}')
def images(breed:str):
    try: label=safe_breed_name(breed)
    except ValueError as exc: raise HTTPException(400,str(exc)) from exc
    folder=(ORIGINAL/label).resolve()
    if folder.parent!=ORIGINAL.resolve(): raise HTTPException(400,'Invalid breed path.')
    if not folder.exists(): return {'breed':label,'images':[]}
    result=[]
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS: result.append({'filename':p.name,'url':f'/files/dataset/original/{label}/{p.name}'})
    return {'breed':label,'images':result}

@router.delete('/images/{breed}/{filename}')
def remove(breed:str,filename:str):
    try:
        result=delete_image(breed,filename)
        from backend.services.activity_service import record_activity
        record_activity('delete',f'Deleted an image from {result["breed"]}.',{'filename':result['filename'],'breed':result['breed']})
        return result
    except (ValueError,FileNotFoundError) as exc: raise HTTPException(404 if isinstance(exc,FileNotFoundError) else 400,str(exc)) from exc

@router.post('/validate')
def validate():
    try: return validate_dataset()
    except DatasetError as exc: raise HTTPException(400,str(exc)) from exc

@router.post('/splits/create')
def splits(force:bool=False):
    try:
        data=create_splits(force=force); return {'splits':{k:len(v) for k,v in data.items()},'manifest':'dataset/splits/manifest.json'}
    except (DatasetError,ValueError) as exc: raise HTTPException(400,str(exc)) from exc
