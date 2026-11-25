"""Realistic training-only augmentation strategies."""
from pathlib import Path
import uuid
import numpy as np
from PIL import Image,ImageDraw
import albumentations as A
from config import AUGMENTED,SAMPLES,SIZE
from utils.io import write_json
from utils.io import read_image

STRATEGIES=('lighting','pose','camera_quality','perspective_scaling','occlusion','combined')

def pipelines():
    return {
      'lighting':A.Compose([A.RandomBrightnessContrast(brightness_limit=.18,contrast_limit=.18,p=.8),A.HueSaturationValue(hue_shift_limit=6,sat_shift_limit=12,val_shift_limit=10,p=.45),A.RandomShadow(p=.2)]),
      'pose':A.Compose([A.HorizontalFlip(p=.5),A.Rotate(limit=15,p=.75),A.Affine(translate_percent=(-.04,.04),scale=(.92,1.08),p=.5)]),
      'camera_quality':A.Compose([A.OneOf([A.GaussianBlur(blur_limit=(3,5)),A.GaussNoise(std_range=(.01,.04)),A.ImageCompression(quality_range=(60,92))],p=.7),A.Sharpen(p=.2)]),
      'perspective_scaling':A.Compose([A.Perspective(scale=(.02,.05),p=.4),A.Affine(scale=(.9,1.12),translate_percent=(-.04,.04),p=.7)]),
      'occlusion':A.Compose([A.CoarseDropout(num_holes_range=(1,4),hole_height_range=(.04,.14),hole_width_range=(.04,.14),fill='random',p=.8)]),
      'combined':A.Compose([A.HorizontalFlip(p=.5),A.Affine(translate_percent=(-.04,.04),scale=(.9,1.1),rotate=(-15,15),p=.7),A.RandomBrightnessContrast(p=.55),A.OneOf([A.GaussianBlur(blur_limit=(3,5)),A.GaussNoise(std_range=(.01,.04))],p=.2),A.CoarseDropout(num_holes_range=(1,3),hole_height_range=(.04,.12),hole_width_range=(.04,.12),fill='random',p=.18)])}

def apply(image,strategy='combined'):
    if strategy not in pipelines(): raise ValueError(f'Unknown strategy: {strategy}')
    arr=np.asarray(image.convert('RGB') if isinstance(image,Image.Image) else image,dtype=np.uint8)
    return pipelines()[strategy](image=arr)['image']

def generate(splits,count=3,strategy='combined',seed=42):
    """Create copies only from the original paths assigned to training."""
    if count<0: raise ValueError('count must be zero or greater')
    chosen=STRATEGIES[:-1] if strategy=='all' else (strategy,)
    if any(s not in STRATEGIES for s in chosen): raise ValueError(f'Unknown strategy {strategy}')
    rng=np.random.default_rng(seed); generated=[]
    for row in splits['train']:
        base=read_image(row['path']).resize(SIZE,Image.Resampling.LANCZOS)
        for i in range(count):
            kind=chosen[int(rng.integers(len(chosen)))]
            result=Image.fromarray(apply(base,kind)); destdir=AUGMENTED/row['label']; destdir.mkdir(parents=True,exist_ok=True)
            dest=destdir/f'{Path(row["path"]).stem}_{i+1}_{uuid.uuid4().hex[:8]}.jpg'; result.save(dest,quality=94)
            generated.append({'path':str(dest),'source':row['path'],'label':row['label'],'strategy':kind})
            if i==0: save_pair(base,result,SAMPLES/f'{row["label"]}_{uuid.uuid4().hex[:8]}.jpg')
    write_json(generated,AUGMENTED/'manifest.json')
    return generated

def save_pair(original,augmented,path):
    path.parent.mkdir(parents=True,exist_ok=True); canvas=Image.new('RGB',(448,250),'white')
    canvas.paste(original.resize((224,224)),(0,26)); canvas.paste(augmented.resize((224,224)),(224,26))
    draw=ImageDraw.Draw(canvas); draw.text((8,6),'Original',fill='black'); draw.text((232,6),'Augmented',fill='black'); canvas.save(path)
