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

## Production deployment (Vercel + Render)

The repository is a two-service app: deploy `frontend/` as a Vercel static site and deploy the FastAPI app from the repository root as a Render web service. The GitHub repository can remain private; connect it through your Render and Vercel GitHub integrations.

### Render backend

Create a **Python web service** from `AP24110011112/breedaugment-ai`, branch `main`, with the repository root as its service root (leave Root Directory blank):

| Setting | Value |
| --- | --- |
| Build command | `pip install -r backend/requirements.txt` |
| Start command | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |
| Health check path | `/api/health` |

Use Python 3.11; `.python-version` pins the Render build to that minor version. Attach a persistent disk mounted at `/var/data`, then set these service environment variables:

```text
BREEDAUGMENT_DATA_DIR=/var/data
FRONTEND_ORIGINS=https://YOUR-VERCEL-SITE.vercel.app
MAX_UPLOAD_BYTES=15728640
```

Replace the Vercel origin with the actual production URL, without a trailing slash. Add a comma-separated list if you also need to allow additional frontend origins. Render's `PORT` is provided by the platform; do not set it manually. Without a persistent disk, uploads, generated images, model files, and reports disappear on service restarts or deploys.

### Vercel frontend

Import the same GitHub repository into Vercel. Set **Root Directory** to `frontend` and use:

| Setting | Value |
| --- | --- |
| Framework preset | Vite |
| Install command | `npm ci` |
| Build command | `npm run build` |
| Output directory | `dist` |

Add this Vercel Production environment variable, replacing the hostname with the Render service URL:

```text
VITE_API_URL=https://YOUR-RENDER-SERVICE.onrender.com
```

Vite embeds `VITE_API_URL` into the frontend during its build, so redeploy Vercel after changing it. The Axios service uses this origin for API requests and generated image links. Then update Render's `FRONTEND_ORIGINS` with the Vercel production URL and redeploy Render. Verify the API at `https://YOUR-RENDER-SERVICE.onrender.com/api/health` and open the Vercel URL.

### Dataset, models, and access

There is no database in this project. Dataset images, trained `.keras` models, and JSON experiment state are stored as files. They are deliberately excluded from Git, so a new hosted deployment starts without the local 7,390 Oxford Pet images or local trained weights. To use prediction immediately, copy the model files and their matching `*_config.json` files to `/var/data/models`; otherwise populate the hosted dataset and train the models there. Do not add these assets or credentials to Git. Keep the original dataset attribution in mind when redistributing images.

The current app has no user authentication: its dataset upload/delete and model training endpoints are callable by anyone who can reach the public API. CORS only controls browser-origin access and is not API authentication. Before leaving the service open to the public, add authentication/authorization for write and training operations or restrict the service to a trusted audience. Dataset upload requests are capped at 100 MB total, and each validated image is capped by `MAX_UPLOAD_BYTES` (15 MB by default).

### Automatic deploys

The Git integrations can watch `main` and automatically rebuild Vercel and redeploy Render after each push. Enable automatic deploys on both service dashboards, and keep the backend and frontend environment variables set in their respective dashboards. Render supports Git-connected private repositories; Vercel likewise needs permission to access this private repository.

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
