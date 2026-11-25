"""Project paths and image settings."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'dataset'; ORIGINAL=DATA/'original'; AUGMENTED=DATA/'augmented'; PROCESSED=DATA/'processed'
MODELS=ROOT/'models'; OUTPUTS=ROOT/'outputs'; SAMPLES=OUTPUTS/'augmentation_samples'; GRAPHS=OUTPUTS/'graphs'; CM=OUTPUTS/'confusion_matrix'; REPORTS=OUTPUTS/'evaluation'
SIZE=(224,224); SEED=42; EXTENSIONS={'.jpg','.jpeg','.png'}
for p in (ORIGINAL,AUGMENTED,PROCESSED,MODELS,SAMPLES,GRAPHS,CM,REPORTS): p.mkdir(parents=True,exist_ok=True)
