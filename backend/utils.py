import torch as t
import numpy as np

MAX_CLASSES = 15

def predict_image(image: list[float], model, device, top_k: int = 10) -> dict:
    # RESHAPE
    # Input comes in as flat 784 list -> (1 batch, 1 channel, 28 height, 28 width)
    x = t.tensor(image, dtype=t.float32).view(1, 1, 28, 28)

    # Invert
    x = 1.0 - x
    # Match the loader.py normalization
    x = (x - 0.1307) / 0.3081
    
    # ROTATE FOR EMNIST
    # The frontend sends an "Upright" image.
    # EMNIST models are trained on (sideways) images. They're still supposed to recognize upright ones,
    # but this pre-rotation helps.
    # We flip the last two dimensions (Height and Width) to match the model's worldview.
    x = x.transpose(-1, -2)

    # Send to GPU if available
    x = x.to(device)

    # So glad I made this
    # # --------------- DEBUG --------------- #
    # # This prints the image to your SERVER TERMINAL so you can see what the model sees.
    # print("\n------ INCOMING IMAGE ------")
    # img_data = x.squeeze().cpu().numpy()
    # for row in img_data:
    #     line = ""
    #     for pixel in row:
    #         # Use distinct chars for different intensity
    #         if pixel > 0.7: line += "@"
    #         elif pixel > 0.3: line += "."
    #         else: line += " "
    #     print(line)
    # print("------------------------------\n")
    # # ------------------------------------ #


    # Ensure top_k is an int within valid range
    top_k = max(1, min(MAX_CLASSES, int(top_k)))

    with t.no_grad():
        logits = model(x)
        probs = t.softmax(logits, dim=1)
        topk = t.topk(probs, k=top_k)
    
    label_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    
    # Move back to CPU for response processing
    indices = topk.indices[0].cpu().numpy()
    values = topk.values[0].cpu().numpy()

    results = [{"char": label_map[i], "prob": float(p)} for i, p in zip(indices, values)]
    
    # Debug print to see if the model is confident or guessing
    print(f"Top prediction: {results[0]['char']} ({results[0]['prob']:.4f})")
    
    return {"predictions": results}
