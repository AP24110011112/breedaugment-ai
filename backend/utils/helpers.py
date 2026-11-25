"""Safe filesystem and image helpers used by API services."""
from pathlib import Path
import json, re, uuid
from PIL import Image, UnidentifiedImageError
from backend.config import MAX_UPLOAD_BYTES

ALLOWED_EXTENSIONS={'.jpg','.jpeg','.png'}

def safe_breed_name(value: str) -> str:
    text=re.sub(r'[^A-Za-z0-9_-]+','_',value.strip()).strip('_-')
    if not text or text in {'.','..'} or len(text)>64: raise ValueError('Enter a breed name using letters, numbers, spaces, hyphens, or underscores (max 64 characters).')
    return text

def safe_filename(value: str) -> str:
    name=Path(value or '').name
    stem=re.sub(r'[^A-Za-z0-9_-]+','_',Path(name).stem).strip('_-')[:100]
    ext=Path(name).suffix.lower()
    if not stem or ext not in ALLOWED_EXTENSIONS: raise ValueError('Only JPG, JPEG, or PNG image files are supported.')
    return f'{stem}_{uuid.uuid4().hex[:8]}{ext}'

def validate_image_bytes(data: bytes, filename: str) -> tuple[int,int]:
    if not data: raise ValueError('The uploaded file is empty.')
    if len(data)>MAX_UPLOAD_BYTES: raise ValueError(f'Image is larger than the {MAX_UPLOAD_BYTES//(1024*1024)} MB upload limit.')
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS: raise ValueError('Only JPG, JPEG, or PNG image files are supported.')
    try:
        from io import BytesIO
        with Image.open(BytesIO(data)) as image:
            image.verify()
        with Image.open(BytesIO(data)) as image:
            return image.width,image.height
    except (OSError,UnidentifiedImageError) as exc: raise ValueError('The uploaded file is not a readable image.') from exc

def load_rgb(path):
    with Image.open(path) as image: image.load(); return image.convert('RGB')

def write_json(data,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(data,indent=2,default=str),encoding='utf-8'); tmp.replace(path)

def read_json(path,default=None):
    path=Path(path)
    try: return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default
    except (OSError,json.JSONDecodeError): return default
