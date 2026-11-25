"""Albumentations previews and training-only augmentation generation."""
from pathlib import Path
from io import BytesIO
import uuid
import re
from datetime import datetime,timezone
from threading import Lock
import numpy as np
from PIL import Image
import albumentations as A
from backend.config import AUGMENTED,DATASET,ORIGINAL,PROCESSED,SAMPLES,SPLITS
from backend.utils.helpers import load_rgb,read_json,write_json,safe_breed_name,validate_image_bytes

_SAVE_LOCK=Lock()

PIPELINES={
 'lighting':A.Compose([A.RandomBrightnessContrast(.18,.18,p=.8),A.HueSaturationValue(6,12,10,p=.45),A.RandomShadow(p=.2)]),
 'rotation':A.Compose([A.HorizontalFlip(p=.5),A.Rotate(limit=15,p=.75)]),
 'perspective':A.Compose([A.Perspective(scale=(.02,.05),p=.7)]),
 'scaling':A.Compose([A.Affine(scale=(.9,1.12),translate_percent=(-.04,.04),rotate=(-5,5),p=.85)]),
 'blur_noise':A.Compose([A.OneOf([A.GaussianBlur(blur_limit=(3,5)),A.GaussNoise(std_range=(.01,.04))],p=.8)]),
 'occlusion':A.Compose([A.CoarseDropout(num_holes_range=(1,4),hole_height_range=(.04,.14),hole_width_range=(.04,.14),fill='random',p=.8)]),
 'combined':A.Compose([A.HorizontalFlip(p=.5),A.Affine(translate_percent=(-.04,.04),scale=(.9,1.1),rotate=(-15,15),p=.7),A.RandomBrightnessContrast(p=.55),A.OneOf([A.GaussianBlur(blur_limit=(3,5)),A.GaussNoise(std_range=(.01,.04))],p=.2),A.CoarseDropout(num_holes_range=(1,3),hole_height_range=(.04,.12),hole_width_range=(.04,.12),fill='random',p=.18)])
}

def transform(image,strategy):
    if strategy not in PIPELINES: raise ValueError(f'Unknown augmentation strategy: {strategy}')
    arr=np.asarray(image.convert('RGB') if isinstance(image,Image.Image) else image,dtype=np.uint8)
    return PIPELINES[strategy](image=arr)['image']

def generate_preview(image_bytes,strategy,count=3):
    if not 1<=count<=20: raise ValueError('Choose between 1 and 20 variants.')
    image=Image.open(BytesIO(image_bytes)).convert('RGB')
    SAMPLES.mkdir(parents=True,exist_ok=True); results=[]
    for _ in range(count):
        name=f'preview_{uuid.uuid4().hex}.jpg'; path=SAMPLES/name
        Image.fromarray(transform(image,strategy)).save(path,quality=94)
        results.append({'filename':name,'url':f'/files/outputs/samples/{name}','strategy':strategy})
    return results

def training_variants(count=2,strategy='combined'):
    if not 0<=count<=20: raise ValueError('Choose between 0 and 20 variants per training image.')
    manifest=read_json(SPLITS/'manifest.json',None)
    if not manifest:
        from backend.services.dataset_service import create_splits
        manifest=create_splits()
    train_sources={str(Path(r['path']).resolve()):r['label'] for r in manifest['train']}
    existing=read_json(AUGMENTED/'manifest.json',[])
    valid=[]
    for item in existing or []:
        source=str(Path(item.get('source','')).resolve())
        aug=Path(item.get('path',''))
        if source in train_sources and train_sources[source]==item.get('label') and aug.is_file() and aug.suffix.lower() in {'.jpg','.jpeg','.png'}:
            valid.append(item)
    if count==0: return []
    by_source={source:[] for source in train_sources}
    for item in valid:
        source=str(Path(item['source']).resolve())
        if source in by_source: by_source[source].append(item)
    needs_generation=any(len(items)<count for items in by_source.values())
    if not needs_generation: return [item for source in train_sources for item in by_source[source][:count]]
    made=[]
    for source,label in train_sources.items():
        items=by_source[source][:count]
        base=None
        while len(items)<count:
            if base is None: base=load_rgb(source)
            name=f'{Path(source).stem}_{uuid.uuid4().hex[:10]}.jpg'; target_dir=AUGMENTED/label; target_dir.mkdir(parents=True,exist_ok=True); target=target_dir/name
            Image.fromarray(transform(base,strategy)).save(target,quality=94)
            items.append({'path':str(target.resolve()),'source':source,'label':label,'strategy':strategy})
        made.extend(items)
    write_json(made,AUGMENTED/'manifest.json')
    return made

def list_samples():
    results=[]
    for folder,base_url in ((SAMPLES,'/files/outputs/samples'),(Path(__file__).resolve().parents[2]/'outputs'/'augmentation_samples','/files/outputs/augmentation_samples')):
        if folder.exists():
            for path in sorted(folder.glob('*.jpg'),key=lambda x:x.stat().st_mtime,reverse=True)[:100]:
                results.append({'filename':path.name,'url':f'{base_url}/{path.name}'})
    return results[:100]

def add_previews_to_dataset(breed,original_filename,images,create_category=False):
    """Persist chosen preview files under dataset/original/<breed> safely."""
    from backend.services.dataset_service import category_summary
    label=safe_breed_name(breed)
    if not images or len(images)>12: raise ValueError('Select between 1 and 12 generated images to add.')
    names=[Path(item.filename).name for item in images]
    if len(set(names))!=len(names): raise ValueError('The selected preview list contains duplicate filenames.')
    payload=[]
    for item,filename in zip(images,names):
        if filename!=item.filename or Path(filename).suffix.lower() not in {'.jpg','.jpeg','.png'}:
            raise ValueError('Invalid preview image filename.')
        if item.strategy not in PIPELINES: raise ValueError('The preview has an unsupported augmentation strategy.')
        source=(SAMPLES/filename).resolve()
        if source.parent!=SAMPLES.resolve() or not source.is_file():
            raise FileNotFoundError(f'Generated preview {filename} is no longer available. Generate it again.')
        data=source.read_bytes(); validate_image_bytes(data,filename)
        payload.append((filename,item.strategy,data))

    destination=(ORIGINAL/label).resolve()
    if destination.parent!=ORIGINAL.resolve(): raise ValueError('Invalid dataset category.')
    if destination.exists() and not destination.is_dir(): raise ValueError('The selected dataset destination is not a folder.')
    if not destination.exists() and not create_category:
        raise FileNotFoundError(f'The dataset folder for {label} does not exist. Choose create category to make it.')

    safe_stem=re.sub(r'[^A-Za-z0-9_-]+','_',Path(original_filename or 'pet-image').stem).strip('_-')[:48] or 'pet-image'
    metadata_path=DATASET/'augmentation_metadata.json'
    created_folder=False; saved=[]
    with _SAVE_LOCK:
        try:
            if not destination.exists():
                destination.mkdir(parents=False)
                created_folder=True
            metadata=read_json(metadata_path,{'images':[]}) or {'images':[]}
            rows=metadata.get('images',[]) if isinstance(metadata,dict) else []
            timestamp=datetime.now(timezone.utc).isoformat()
            for preview_name,strategy,data in payload:
                while True:
                    output_name=f'aug_{safe_stem}_{uuid.uuid4().hex[:12]}{Path(preview_name).suffix.lower()}'
                    target=destination/output_name
                    try:
                        with target.open('xb') as out: out.write(data)
                        break
                    except FileExistsError:
                        continue
                saved.append(target)
                rows.append({'filename':output_name,'breed':label,'original_filename':Path(original_filename or '').name,
                             'preview_filename':preview_name,'strategy':strategy,'added_at':timestamp})
            write_json({'images':rows},metadata_path)
            (SPLITS/'manifest.json').unlink(missing_ok=True)
        except Exception:
            for path in saved: path.unlink(missing_ok=True)
            if created_folder:
                try: destination.rmdir()
                except OSError: pass
            raise

    summary=category_summary(label)
    from backend.services.activity_service import record_activity
    record_activity('augmentation',f'Added {len(saved)} augmented image(s) to {label}.',
                    {'breed':label,'count':len(saved),'original_filename':Path(original_filename or '').name})
    return {'added':len(saved),'breed':label,'images':[p.name for p in saved],**summary}
