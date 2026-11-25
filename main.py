"""CLI: python main.py --stage all"""
import argparse,shutil
from pathlib import Path
from config import ORIGINAL,AUGMENTED,PROCESSED,MODELS,REPORTS
from src.dataset import inspect_dataset,prepare_dataset
from src.augment import generate
from src.validate import validate

def run(stage,count,strategy,epochs,batch,pretrained,seed):
    summary,rows=inspect_dataset(); print('Original images by breed:',summary)
    splits=prepare_dataset(seed,force=(stage in ('preprocess','all'))); classes=sorted(summary)
    if stage in ('augment','all'):
        generated=generate(splits,count,strategy,seed); print('Generated:',len(generated))
        print('Validation:',validate(generated)['valid'],'valid images')
    if stage=='validate':
        from utils.io import read_json
        manifest=read_json(AUGMENTED/'manifest.json',[]); print(validate(manifest))
    if stage in ('train-original','all'):
        from src.train import train
        train(PROCESSED/'train',PROCESSED/'validation',classes,'original',epochs,batch,pretrained)
    if stage in ('train-augmented','all'):
        from src.train import train
        train_dir=PROCESSED/'train_augmented'; shutil.rmtree(train_dir,ignore_errors=True); shutil.copytree(PROCESSED/'train',train_dir)
        for breed_dir in AUGMENTED.iterdir():
            if breed_dir.is_dir():
                target=train_dir/breed_dir.name; target.mkdir(parents=True,exist_ok=True)
                for image in breed_dir.glob('*.jpg'): shutil.copy2(image,target/image.name)
        train(train_dir,PROCESSED/'validation',classes,'augmented',epochs,batch,pretrained)
    if stage in ('evaluate','all'):
        from src.evaluate import evaluate,compare
        original=evaluate(MODELS/'original.keras',PROCESSED/'test',classes,'original')
        augmented=evaluate(MODELS/'augmented.keras',PROCESSED/'test',classes,'augmented')
        print(compare(original,augmented).to_string(index=False))

def main():
    p=argparse.ArgumentParser(description='BreedAugment AI workflow')
    p.add_argument('--stage',choices=['check','preprocess','augment','validate','train-original','train-augmented','evaluate','all'],default='all')
    p.add_argument('--augment-count',type=int,default=3); p.add_argument('--strategy',choices=['combined','lighting','pose','camera_quality','perspective_scaling','occlusion','all'],default='combined')
    p.add_argument('--epochs',type=int,default=12); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--seed',type=int,default=42); p.add_argument('--no-pretrained',action='store_true')
    a=p.parse_args()
    if a.stage=='check':
        try: print(inspect_dataset()[0])
        except Exception as e: p.error(str(e))
        return
    try: run(a.stage,a.augment_count,a.strategy,a.epochs,a.batch_size,not a.no_pretrained,a.seed)
    except Exception as e: p.error(str(e))
if __name__=='__main__': main()
