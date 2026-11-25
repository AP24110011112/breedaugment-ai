"""BreedAugment AI multi-page Streamlit application."""
from pathlib import Path
import json,shutil
import streamlit as st
from PIL import Image
from config import ORIGINAL,AUGMENTED,PROCESSED,MODELS,OUTPUTS,REPORTS,SAMPLES
from utils.io import list_images,read_json
from src.dataset import inspect_dataset,prepare_dataset
from src.augment import apply,generate,STRATEGIES
from src.validate import validate

st.set_page_config(page_title='BreedAugment AI',page_icon='🐾',layout='wide')
st.title('🐾 BreedAugment AI')
st.caption('Pet breed identification and training-only image augmentation')
page=st.sidebar.radio('Sections',['Home Dashboard','Dataset Management','Image Augmentation','Augmentation Comparison','Model Training','Model Evaluation','Pet Breed Prediction','About Project'])
try: summary,records=inspect_dataset(); data_error=None
except Exception as exc: summary,records={},[]; data_error=str(exc)

def augment_train_images(count,strategy):
    split=prepare_dataset()
    made=generate(split,count,strategy)
    return validate(made),made

if page=='Home Dashboard':
    st.header('Home Dashboard')
    if records:
        cols=st.columns(4); cols[0].metric('Original photos',len(records)); cols[1].metric('Augmented photos',len(list_images(AUGMENTED))); cols[2].metric('Breed classes',len(summary)); cols[3].metric('Training status','Trained' if list(MODELS.glob('*.keras')) else 'Pending')
        st.bar_chart({breed:{'readable':info['readable'],'corrupt':info['corrupt']} for breed,info in summary.items()})
    else: st.info(data_error)
    comparison=REPORTS/'model_comparison.csv'
    if comparison.exists():
        import pandas as pd; st.dataframe(pd.read_csv(comparison),use_container_width=True)
    st.write('Controlled, realistic augmentation can help a classifier handle variation, but accuracy improvement is not guaranteed. Compare on untouched test originals.')
elif page=='Dataset Management':
    st.header('Dataset Management')
    st.code('dataset/original/\n├── Breed_A/\n│   ├── photo_01.jpg\n└── Breed_B/\n    └── photo_02.png')
    if records:
        import pandas as pd; st.dataframe(pd.DataFrame([{'Breed':k,**v} for k,v in summary.items()]),use_container_width=True)
        if st.button('Check and prepare original-first splits'):
            try: prepare_dataset(force=True); st.success('Created resized train, validation, and test splits. Augmentation has not touched validation or test photos.')
            except Exception as exc: st.error(str(exc))
    else: st.warning(data_error)
elif page=='Image Augmentation':
    st.header('Image Augmentation')
    up=st.file_uploader('Preview a pet photo',type=['jpg','jpeg','png']); strategy=st.selectbox('Strategy',STRATEGIES)
    count=st.number_input('Variants per training photo',min_value=1,max_value=30,value=3)
    if up:
        try:
            image=Image.open(up).convert('RGB'); changed=Image.fromarray(apply(image,strategy)); a,b=st.columns(2); a.image(image,caption='Original',use_container_width=True); b.image(changed,caption='Augmented preview',use_container_width=True)
        except Exception as exc: st.error(str(exc))
    if st.button('Generate training augmentations'):
        try:
            if not records: raise ValueError(data_error)
            report,made=augment_train_images(int(count),strategy); st.success(f'Generated {len(made)} images. Validation found {report["issues"]} issue(s).')
        except Exception as exc: st.error(str(exc))
elif page=='Augmentation Comparison':
    st.header('Augmentation Comparison')
    files=list_images(SAMPLES)
    if files:
        selected=st.selectbox('Saved comparison',files,format_func=lambda p:p.name); st.image(selected,use_container_width=True)
    else: st.info('Generate augmentations to save side-by-side examples.')
elif page=='Model Training':
    st.header('Model Training')
    st.write('Both models use the same original-first split, validation photos, MobileNetV2 architecture, and training parameters.')
    target=st.radio('Training set',['Original only','Original plus augmented']); epochs=st.slider('Maximum epochs',1,50,12); batch=st.selectbox('Batch size',[8,16,32,64],index=2); pretrained=st.checkbox('Initialize with ImageNet weights',True)
    if st.button('Train selected model'):
        try:
            if not records: raise ValueError(data_error)
            splits=prepare_dataset(); classes=sorted(summary)
            if target=='Original only': train_dir=PROCESSED/'train'; name='original'
            else:
                if not list_images(AUGMENTED): raise ValueError('Generate augmentations first.')
                train_dir=PROCESSED/'train_augmented'; shutil.rmtree(train_dir,ignore_errors=True); shutil.copytree(PROCESSED/'train',train_dir)
                for breed in AUGMENTED.iterdir():
                    if breed.is_dir():
                        dest=train_dir/breed.name; dest.mkdir(parents=True,exist_ok=True)
                        for photo in list_images(breed): shutil.copy2(photo,dest/photo.name)
                name='augmented'
            from src.train import train
            with st.spinner('Training model…'):
                history=train(train_dir,PROCESSED/'validation',classes,name,int(epochs),int(batch),pretrained)
            st.success(f'{name.title()} model training completed.'); st.line_chart({k:v for k,v in history.items() if k in ('accuracy','val_accuracy','loss','val_loss')})
        except Exception as exc: st.error(str(exc))
elif page=='Model Evaluation':
    st.header('Model Evaluation')
    if (REPORTS/'model_comparison.csv').exists():
        import pandas as pd; st.dataframe(pd.read_csv(REPORTS/'model_comparison.csv'),use_container_width=True)
        for img in sorted((OUTPUTS/'graphs').glob('*.png')): st.image(img,caption=img.name,use_container_width=True)
        for img in sorted((OUTPUTS/'confusion_matrix').glob('*.png')): st.image(img,caption=img.name,use_container_width=True)
    else: st.info('Evaluation pending. Train both models, then use the CLI: python main.py --stage evaluate')
    report=read_json(REPORTS/'augmentation_validation.json')
    if report: st.json(report)
elif page=='Pet Breed Prediction':
    st.header('Pet Breed Prediction')
    models=list(MODELS.glob('*.keras')); upload=st.file_uploader('Upload a pet image',type=['jpg','jpeg','png'])
    if not models: st.info('Train a model before predicting.')
    elif upload:
        chosen=st.selectbox('Model',models,format_func=lambda p:p.stem)
        try:
            from src.predict import load_bundle,predict
            model,cfg=load_bundle(chosen); image=Image.open(upload).convert('RGB'); st.image(image,width=330)
            preds=predict(model,image,cfg['class_names']); st.success(f'{preds[0]["breed"]} · {preds[0]["confidence"]:.1%} confidence')
            st.dataframe([{'Breed':x['breed'],'Confidence':f'{x["confidence"]:.1%}'} for x in preds],use_container_width=True)
            st.caption('Predictions are limited to the breeds present in the trained dataset.')
        except Exception as exc: st.error(f'Prediction failed: {exc}')
elif page=='About Project':
    st.header('About BreedAugment AI')
    st.write('A reproducible college-level computer vision project that tests whether controlled training-image augmentation improves pet breed classification. It makes no guarantee of higher accuracy.')
    st.markdown('**Workflow:** original photos → train/validation/test split → training-only augmentation → two comparable MobileNetV2 models → shared untouched test set.')
