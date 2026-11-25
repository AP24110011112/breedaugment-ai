"""Discover, validate, resize, and split breed folders before augmentation."""
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from PIL import Image
from config import ORIGINAL,PROCESSED,SIZE,SEED
from utils.io import list_images,read_image,write_json,read_json

class DatasetError(ValueError): pass

def inspect_dataset(root=ORIGINAL):
    root=Path(root)
    if not root.is_dir(): raise DatasetError(f'Dataset directory not found: {root}')
    classes=sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.'))
    if not classes: raise DatasetError(f'No breed folders in {root}. Create one folder per label and add JPG/JPEG/PNG photos.')
    rows=[]; summary={}
    for cls in classes:
        valid=0; corrupt=0
        for path in list_images(cls):
            try:
                im=read_image(path); rows.append({'path':str(path.resolve()),'label':cls.name,'width':im.width,'height':im.height}); valid+=1
            except ValueError: corrupt+=1
        summary[cls.name]={'readable':valid,'corrupt':corrupt}
    if not rows: raise DatasetError('No readable JPG, JPEG, or PNG photos were found.')
    return summary,rows

def create_splits(rows,seed=SEED,test_fraction=.15,val_fraction=.15):
    """Partition original file records first; each source is assigned to exactly one split."""
    if len(rows)<3: raise DatasetError('At least 3 readable photos are needed for train/validation/test.')
    import numpy as np
    indices=np.arange(len(rows)); labels=[r['label'] for r in rows]
    can_stratify=min(Counter(labels).values())>=3
    try:
        train_ids,test_ids=train_test_split(indices,test_size=test_fraction,random_state=seed,stratify=labels if can_stratify else None)
        train_labels=[labels[i] for i in train_ids]
        stratify=train_labels if min(Counter(train_labels).values())>=2 else None
        train_ids,val_ids=train_test_split(train_ids,test_size=val_fraction/(1-test_fraction),random_state=seed,stratify=stratify)
    except ValueError as exc: raise DatasetError(f'Could not split the images: {exc}. Add more photos per breed.') from exc
    return {k:[rows[i] for i in ids] for k,ids in [('train',train_ids),('validation',val_ids),('test',test_ids)]}

def prepare_dataset(seed=SEED,force=False):
    manifest=PROCESSED/'split_manifest.json'
    if manifest.exists() and not force:
        saved=read_json(manifest)
        sources=[r['path'] for part in saved.values() for r in part]
        current={str(Path(x['path']).resolve()) for x in inspect_dataset()[1]}
        if current==set(sources): return saved
    _,rows=inspect_dataset(); splits=create_splits(rows,seed)
    for split,records in splits.items():
        out=PROCESSED/split
        import shutil; shutil.rmtree(out,ignore_errors=True)
        for rec in records:
            dest=out/rec['label']/(Path(rec['path']).stem+'.jpg'); dest.parent.mkdir(parents=True,exist_ok=True)
            read_image(rec['path']).resize(SIZE,Image.Resampling.LANCZOS).save(dest,quality=95)
    write_json(splits,manifest); return splits
