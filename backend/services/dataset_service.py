"""Dataset inventory, upload, deletion, validation, and leakage-safe splits."""
from pathlib import Path
from collections import Counter
from threading import Lock
from PIL import Image
from backend.config import DATASET,ORIGINAL,PROCESSED,SPLITS,MODEL_INPUT_SIZE
from backend.utils.helpers import ALLOWED_EXTENSIONS,load_rgb,read_json,write_json,safe_breed_name,safe_filename,validate_image_bytes

class DatasetError(ValueError): pass

_AUGMENTATION_METADATA=DATASET/'augmentation_metadata.json'
_metadata_lock=Lock()

def _metadata_rows():
    stored=read_json(_AUGMENTATION_METADATA,{'images':[]}) or {'images':[]}
    return stored.get('images',[]) if isinstance(stored,dict) else []

def _augmented_names(breed):
    folder=(ORIGINAL/breed).resolve()
    names=set()
    for item in _metadata_rows():
        if item.get('breed')!=breed: continue
        name=Path(item.get('filename','')).name
        path=(folder/name).resolve()
        if path.parent==folder and path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
            names.add(name)
    return names

def category_summary(breed):
    label=safe_breed_name(breed); folder=ORIGINAL/label
    images=[p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS] if folder.is_dir() else []
    augmented=len(_augmented_names(label))
    return {'breed':label,'total_images':len(images),'original_images':max(0,len(images)-augmented),'augmented_images':augmented,'destination':str(folder.resolve())}

def inspect_dataset():
    if not ORIGINAL.exists(): raise DatasetError('Dataset/original folder is missing.')
    breeds=sorted(p for p in ORIGINAL.iterdir() if p.is_dir() and not p.name.startswith('.'))
    result=[]; totals={'images':0,'corrupt':0,'classes':len(breeds)}
    for breed in breeds:
        count=0; corrupt=0
        for path in sorted(breed.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in ALLOWED_EXTENSIONS: continue
            try:
                with Image.open(path) as im: im.verify()
                count+=1
            except Exception: corrupt+=1
        augmented=len(_augmented_names(breed.name))
        result.append({'breed':breed.name,'count':count,'original_count':max(0,count-augmented),'augmented_count':augmented,'corrupt':corrupt})
        totals['images']+=count; totals['corrupt']+=corrupt
    totals['breeds']=result
    return totals

def get_breeds(): return inspect_dataset()['breeds']

def create_breed_category(name):
    label=safe_breed_name(name); target=ORIGINAL/label; created=not target.exists(); target.mkdir(parents=True,exist_ok=True)
    if created: (SPLITS/'manifest.json').unlink(missing_ok=True)
    return {'breed':label,'created':created,'count':sum(1 for p in target.rglob('*') if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS)}

def upload_images(breed,files):
    label=safe_breed_name(breed); directory=ORIGINAL/label; directory.mkdir(parents=True,exist_ok=True)
    saved=[]
    for filename,data in files:
        width,height=validate_image_bytes(data,filename)
        target=directory/safe_filename(filename)
        target.write_bytes(data)
        saved.append({'breed':label,'filename':target.name,'path':f'/files/dataset/original/{label}/{target.name}','width':width,'height':height})
    return saved

def delete_image(breed,filename):
    label=safe_breed_name(breed); name=Path(filename).name
    target=(ORIGINAL/label/name).resolve()
    if target.parent!= (ORIGINAL/label).resolve(): raise ValueError('Invalid image path.')
    if not target.is_file() or target.suffix.lower() not in ALLOWED_EXTENSIONS: raise FileNotFoundError('Image was not found.')
    target.unlink()
    with _metadata_lock:
        metadata=_metadata_rows()
        remaining=[item for item in metadata if not (item.get('breed')==label and item.get('filename')==name)]
        if len(remaining)!=len(metadata): write_json({'images':remaining},_AUGMENTATION_METADATA)
    if not any(item.is_file() and item.suffix.lower() in ALLOWED_EXTENSIONS for item in target.parent.iterdir()):
        try: target.parent.rmdir()
        except OSError: pass
    # Force a fresh split before the next training/evaluation operation.
    (SPLITS/'manifest.json').unlink(missing_ok=True)
    return {'deleted':True,'breed':label,'filename':name}

def validate_dataset():
    stats=inspect_dataset(); report={'total_images':stats['images'],'corrupt_images':stats['corrupt'],'class_count':stats['classes'],'breeds':stats['breeds'],'valid':stats['corrupt']==0}
    write_json(report,SPLITS/'dataset_validation.json'); return report

def create_splits(force=False,seed=42):
    stats=inspect_dataset(); images=[]
    for breed_row in stats['breeds']:
        folder=ORIGINAL/breed_row['breed']
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
                try:
                    with Image.open(path) as im: im.verify()
                    images.append({'path':str(path.resolve()),'label':breed_row['breed']})
                except Exception: continue
    if len(images)<3: raise DatasetError('Add at least three readable images before creating train/validation/test splits.')
    manifest_path=SPLITS/'manifest.json'; current={r['path'] for r in images}
    if manifest_path.exists() and not force:
        existing=read_json(manifest_path,{})
        found={r['path'] for rows in existing.values() for r in rows}
        if found==current and set(existing)=={'train','validation','test'} and all((PROCESSED/k).exists() for k in existing): return existing
    # Reuse an existing verified project split where possible to preserve the current experiment.
    old=read_json(PROCESSED/'split_manifest.json',None)
    if not force and old and {r['path'] for rows in old.values() for r in rows}==current:
        splits=old
    else:
        from sklearn.model_selection import train_test_split
        import numpy as np
        indices=np.arange(len(images)); labels=[r['label'] for r in images]; counts=Counter(labels)
        strat=labels if min(counts.values())>=3 else None
        try:
            train_idx,test_idx=train_test_split(indices,test_size=.15,random_state=seed,stratify=strat)
            train_labels=[labels[i] for i in train_idx]; strat_train=train_labels if min(Counter(train_labels).values())>=2 else None
            train_idx,val_idx=train_test_split(train_idx,test_size=.15/.85,random_state=seed,stratify=strat_train)
        except ValueError as exc: raise DatasetError(f'Could not make a leakage-safe split: {exc}. Add images to each breed.') from exc
        splits={name:[images[i] for i in idx] for name,idx in [('train',train_idx),('validation',val_idx),('test',test_idx)]}
    import shutil
    for split,rows in splits.items():
        target=PROCESSED/split
        expected=[target/row['label']/(Path(row['path']).stem+'.jpg') for row in rows]
        # A prior run may have stopped halfway through writing a split. Check
        # the expected cache entries so an existing but partial directory is
        # rebuilt instead of being mistaken for a complete preprocessing run.
        cache_incomplete=(not target.is_dir() or any(not p.is_file() or p.stat().st_size==0 for p in expected))
        if force or cache_incomplete:
            shutil.rmtree(target,ignore_errors=True)
            for row in rows:
                out=target/row['label']/(Path(row['path']).stem+'.jpg'); out.parent.mkdir(parents=True,exist_ok=True)
                load_rgb(row['path']).resize(MODEL_INPUT_SIZE,Image.Resampling.LANCZOS).save(out,quality=94)
    write_json(splits,manifest_path)
    return splits

def dataset_stats():
    stats=inspect_dataset()
    manifest=read_json(SPLITS/'manifest.json',None)
    if manifest: stats['splits']={k:len(v) for k,v in manifest.items()}
    return stats
