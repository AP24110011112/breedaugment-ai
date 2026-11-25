"""Safe image and JSON helpers."""
from pathlib import Path
import json
from PIL import Image, UnidentifiedImageError
from config import EXTENSIONS

def list_images(folder):
    folder=Path(folder)
    return sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS) if folder.exists() else []

def read_image(path):
    try:
        with Image.open(path) as image:
            image.load(); return image.convert('RGB')
    except (OSError,UnidentifiedImageError) as exc:
        raise ValueError(f'Cannot read image: {path}') from exc

def write_json(data,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,default=str),encoding='utf-8')

def read_json(path,default=None):
    path=Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default
