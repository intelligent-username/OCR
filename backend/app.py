import torch as t
import numpy as np

from pydantic import BaseModel

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from model import EMNIST_VGG

import os

app = FastAPI()

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
    # RESHAPE
    # Input comes in as flat 784 list -> (1 batch, 1 channel, 28 height, 28 width)
    x = t.tensor(req.image, dtype=t.float32).view(1, 1, 28, 28)

    # Match the loader.py normalization
    x = (x - 0.1307) / 0.3081
    
    # ROTATE FOR EMNIST
    # The frontend sends an "Upright" image.
    # EMNIST models are trained on "Transposed" (sideways) images.
    # We flip the last two dimensions (Height and Width) to match the model's worldview.
    x = x.transpose(-1, -2)

    # Send to GPU if available
    x = x.to(device)

    # --- DEBUG: ASCII ART GENERATOR ---
    # This prints the image to your SERVER TERMINAL so you can see what the model sees.
    print("\n--- INCOMING IMAGE DEBUG ---")
    img_data = x.squeeze().cpu().numpy()
    for row in img_data:
        line = ""
        for pixel in row:
            # Use distinct chars for different intensity
            if pixel > 0.7: line += "@"
            elif pixel > 0.3: line += "."
            else: line += " "
        print(line)
    print("------------------------------\n")
    # ----------------------------------

    with t.no_grad():
        logits = model(x)
        probs = t.softmax(logits, dim=1)
        topk = t.topk(probs, k=10)
    
    label_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    
    # Move back to CPU for response processing
    indices = topk.indices[0].cpu().numpy()
    values = topk.values[0].cpu().numpy()

    results = [{"char": label_map[i], "prob": float(p)} for i, p in zip(indices, values)]
    
    # Debug print to see if the model is confident or guessing
    print(f"Top prediction: {results[0]['char']} ({results[0]['prob']:.4f})")
    
    return {"predictions": results}
