import torch as t
import asyncio
import os

from backend.utils import predict_image
from backend.model import EMNIST_VGG

from pydantic import BaseModel

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from huggingface_hub import hf_hub_download

# ----------------------------
# LIFESPAN stuff (basically for startup/shutdown since we run with --reload)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    except asyncio.CancelledError:
        print("Code likely edited, restarting server...")
        return # Suppressing annoying tracebacks on --reload
    except Exception:
        # real startup/shutdown failure
        raise

# LIFESPAN END
# ----------------------------

app = FastAPI(lifespan=lifespan)

device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(f"Server running on: {device}")


# ----------------------------
# MODEL DOWNLOAD

# 1. Setup paths
REPO_ID = "compendious/EMNIST-OCR-WEIGHTS/"
FILENAME = "EMNIST_CNN.pth"

# NOTE: If I ever make this repo private, I need to add authentication tokens to hf_hub_download calls.

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
MODEL_PATH = os.path.join(BASE_DIR, FILENAME)

if not os.path.exists(MODEL_PATH):
    print(f"Model weights not found. Downloading from {REPO_ID}...")
    # This downloads the file and returns the local path
    downloaded_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    
    # Optional: Move it to your backend folder so your existing load logic works
    import shutil
    shutil.copy(downloaded_path, MODEL_PATH)
    print(f"Weights secured at {MODEL_PATH}")

# MODEL DOWNLOAD END
# ----------------------------


# Instantiate the empty architecture first
model = EMNIST_VGG(num_classes=62).to(device)

# Load the weights safely
# Note: If this fails, it means your file is still the old "full model" format.
# If so, re-run your training script to generate a clean state_dict.
try:
    model.load_state_dict(t.load(MODEL_PATH, map_location=device, weights_only=True))
except Exception as e:
    print("State dict load failed, trying legacy full-model load:", e)
    model = t.load(MODEL_PATH, map_location=device, weights_only=False)

model.eval()

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def read_index():
    path = os.path.join("frontend", "index.html")
    return FileResponse(path)

class PredictRequest(BaseModel):
    image: list[float]  # flat 28*28 array
    k: int = 10  # number of top predictions to return

@app.post("/predict")
def predict(req: PredictRequest):
    print(f"Predicting... +{1+1}")
    # top_k currently set to 10 to preserve existing behavior
    return predict_image(req.image, model, device, top_k=req.k)
