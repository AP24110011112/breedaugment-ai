"""Backend paths and runtime settings."""
from pathlib import Path
import os, platform
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.getenv('BREEDAUGMENT_DATA_DIR', str(ROOT))).expanduser().resolve()
os.environ.setdefault('MPLCONFIGDIR', str(DATA_ROOT / '.cache' / 'matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(DATA_ROOT / '.cache'))
DATASET = DATA_ROOT / 'dataset'
ORIGINAL = DATASET / 'original'
AUGMENTED = DATASET / 'augmented'
PROCESSED = DATASET / 'processed'
SPLITS = DATASET / 'splits'
MODELS = DATA_ROOT / 'models'
OUTPUTS = DATA_ROOT / 'outputs'
SAMPLES = OUTPUTS / 'samples'
REPORTS = OUTPUTS / 'reports'
GRAPHS = OUTPUTS / 'graphs'
CONFUSION = OUTPUTS / 'confusion_matrix'
MODEL_INPUT_SIZE = (224, 224)
MAX_UPLOAD_BYTES = int(os.getenv('MAX_UPLOAD_BYTES', 15 * 1024 * 1024))
HOST_ARCH = platform.machine().lower()
IS_APPLE_SILICON = platform.system() == 'Darwin' and HOST_ARCH in {'arm64', 'aarch64'}
IS_MAC_INTEL = platform.system() == 'Darwin' and HOST_ARCH in {'x86_64', 'amd64'}
for directory in (DATA_ROOT / '.cache' / 'matplotlib', ORIGINAL, AUGMENTED, PROCESSED, SPLITS, MODELS, SAMPLES, REPORTS, GRAPHS, CONFUSION):
    directory.mkdir(parents=True, exist_ok=True)
