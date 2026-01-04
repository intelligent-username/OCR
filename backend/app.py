import torch as t
import asyncio
import os

from utils import predict_image

from pydantic import BaseModel

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from model import EMNIST_VGG

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


app = FastAPI(lifespan=lifespan)

device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(f"Server running on: {device}")

# Instantiate the empty architecture first
model = EMNIST_VGG(num_classes=62).to(device)

# Load the weights safely
# Note: If this fails, it means your file is still the old "full model" format.
# If so, re-run your training script to generate a clean state_dict.
try:
    model.load_state_dict(t.load("EMNIST_CNN.pth", map_location=device, weights_only=True))
except Exception as e:
    print("State dict load failed, trying legacy full-load (not recommended for long term):", e)
    model = t.load("../models/EMNIST_CNN.pth", map_location=device, weights_only=False)

model.eval()

app.mount("/static", StaticFiles(directory="../frontend"), name="static")

@app.get("/")
async def read_index():
    path = os.path.join("..", "frontend", "index.html")
    return FileResponse(path)

class PredictRequest(BaseModel):
    image: list[float]  # flat 28*28 array

@app.post("/predict")
def predict(req: PredictRequest):
    print(f"Predicting... +{1+1}")
    return predict_image(req.image, model, device)
