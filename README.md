# BreedAugment AI

A local-first full-stack pet-breed classification and data-augmentation application. React/Vite provides the interface, FastAPI exposes REST endpoints, and TensorFlow/Keras trains and evaluates MobileNetV2 models. Dataset and model files remain on the local machine.

The local workspace uses the Oxford-IIIT Pet dataset: 7,390 images in 37 breed classes. Dataset images, trained weights, and generated outputs are intentionally excluded from Git; they are large and the photographs remain under their original owners' copyright. Dataset source, attribution, and license information are in `dataset/original/DATASET_INFO.txt`. After cloning, add your dataset under `dataset/original/<breed>/` or populate it through Dataset Management. The dashboard does not show fabricated performance: model metrics remain pending until both models are trained and evaluated.

## Architecture

```text
React + Vite (5173)
      │ Axios / REST
      ▼
FastAPI + Uvicorn (8000)
  ├── Dataset, upload, split, validation services
  ├── Albumentations preview and train-only augmentation
  ├── Single background TensorFlow training job + progress polling
  ├── Shared untouched test evaluation
  └── Saved model prediction
      │
      ├── dataset/original, augmented, processed, splits
      ├── models/*.keras + JSON model configuration
      └── outputs/reports, graphs, samples, confusion matrices
```

## macOS setup

Use Python 3.11 and Node.js 20 or newer. Keep the existing workspace `.venv`; activate it before installing backend dependencies.

```bash
cd breedaugment-ai
source .venv/bin/activate
cd backend
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow dependency markers select a compatible wheel for the Mac architecture. This workspace is Apple Silicon (`arm64`) and already has TensorFlow 2.19.1; the requirements allow the arm64 TensorFlow 2.19–2.21 line. Intel macOS is pinned below TensorFlow 2.17 because the TensorFlow pip wheel list currently ends at 2.16 for x86_64. Apple Silicon uses the regular `tensorflow` package for TensorFlow 2.13+, not the obsolete `tensorflow-macos` package. TensorFlow runs on CPU by default; `tensorflow-metal` is optional and should only be installed if its current compatibility matches the selected TensorFlow version. [TensorFlow macOS installation matrix](https://www.tensorflow.org/install/pip), [Apple TensorFlow Metal setup](https://developer.apple.com/metal/tensorflow-plugin/).

Install frontend packages in a second terminal:

```bash
cd breedaugment-ai/frontend
npm install
```

## Start the application

Keep these in two separate VS Code terminals.

**Terminal 1 — backend**

```bash
cd breedaugment-ai/backend
source ../.venv/bin/activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

API health: http://127.0.0.1:8000/api/health

Interactive OpenAPI docs: http://127.0.0.1:8000/docs

**Terminal 2 — frontend**

```bash
cd breedaugment-ai/frontend
npm run dev
```

Open http://localhost:5173. Vite proxies `/api` and `/files` requests to the local FastAPI service. `VITE_API_URL` can override the API origin in a local `.env` file when needed.

## Dataset and experiment design

Breed classes are discovered from `dataset/original/<breed>/`. The bundled dataset is already arranged in that format. Uploads through Dataset Management are validated as readable JPG/JPEG/PNG files and assigned a safe breed folder name. A deterministic stratified split is materialized at `dataset/processed/{train,validation,test}` and recorded at `dataset/splits/manifest.json`.

The split happens before training augmentation. Model A uses only original train images. Model B adds augmentations only when the augmentation manifest links them to a path in the current training split. Validation and test images stay original and are shared by both models. Augmentation previews can be saved into a selected `dataset/original/<breed>/` category; those additions are tracked in `dataset/augmentation_metadata.json` and invalidate the old split so the next experiment rebuilds it.

The Augmentation page creates previews under `outputs/samples/`; previews from an uploaded image are never silently added to training. Model B training makes controlled variants from training originals if no valid training-only variants are available.

## Frontend sections

- **Dashboard:** dataset counts/distribution, model training state and accuracy, latest prediction.
- **Dataset:** upload images into a breed category, browse and delete files, validate images.
- **Augmentation:** choose a strategy and create downloadable image previews.
- **Model training:** start Model A or Model B, poll epoch progress, review accuracy/loss curves.
- **Evaluation:** run and inspect accuracy, weighted precision/recall/F1, classification reports, comparison charts, confusion matrices.
- **Prediction:** upload an image and receive the top three breeds supported by the selected trained model.
- **About:** workflow, objectives, tools, integrity notes.

## REST API

- `GET /api/health`
- `GET /api/dashboard/stats`
- `GET /api/dataset/breeds`
- `GET /api/dataset/stats`
- `GET /api/dataset/images/{breed}`
- `POST /api/dataset/upload` (multipart fields: `breed`, `files`)
- `DELETE /api/dataset/images/{breed}/{filename}`
- `POST /api/dataset/validate`
- `POST /api/dataset/splits/create`
- `POST /api/augmentation/generate` (multipart: `image`, `strategy`, `count`)
- `GET /api/augmentation/samples`
- `POST /api/augmentation/add-to-dataset` (JSON: breed, selected preview filenames/strategies, source filename, and optional category creation)
- `POST /api/training/start`
- `GET /api/training/status`
- `GET /api/evaluation/results`
- `POST /api/evaluation/run`
- `POST /api/prediction/predict` (multipart: `image`, optional `model_type`)
- `GET /api/models/status`

## TensorFlow models

Both model variants use MobileNetV2, 224×224 RGB images, ImageNet initialization, global-average pooling, dropout, a dynamically sized softmax output, Adam, and sparse categorical cross-entropy. Training uses ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau. A single training task is allowed at a time; progress is written to `outputs/reports/training_status.json` and exposed through polling.

Trained models are saved under `models/`. Evaluation artifacts go into `outputs/reports/`, `outputs/graphs/`, and `outputs/confusion_matrix/`. Class labels and training settings are saved beside each `.keras` file.

## Troubleshooting

- **Frontend says API unavailable:** start the backend first; check http://127.0.0.1:8000/api/health.
- **Port busy:** stop the existing process or use another Uvicorn/Vite port and update the Vite proxy.
- **TensorFlow wheel install fails on Intel:** make sure the active interpreter is x86_64 Python 3.11; the requirements intentionally avoid unsupported Intel macOS TensorFlow releases.
- **ImageNet weights cannot download:** retry with network access, or change the frontend training request to `pretrained: false` (the UI currently defaults to ImageNet).
- **No prediction available:** train at least one model first.
- **No metrics shown:** train both Model A and Model B, then click Run evaluation.

## Dataset acknowledgment and license

Oxford-IIIT Pet was authored by Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman, and C. V. Jawahar. Cite: “Cats and Dogs,” CVPR 2012. The Oxford page lists CC BY-SA 4.0 and notes that image copyright remains with original owners. Keep attribution and review license terms before redistributing images or derivatives. See `dataset/original/DATASET_INFO.txt`, the [Oxford dataset page](https://www.robots.ox.ac.uk/~vgg/data/pets/), and the [Zenodo record](https://doi.org/10.5281/zenodo.8067751).

## Future enhancements

Optional TensorFlow Metal acceleration, experiment history UI, class-balanced sampling, configurable augmentation policies, calibration, and exportable model cards.
