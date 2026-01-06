# Optical Character Recognition

![Demonstration](https://varak.dev/host/EMNIST-OCR-DEMO.webp)

Use the tool live [here](https://ocr.varak.dev).

This is an optical character recognition (OCR) tool that extracts characters from drawings. It's made with FastAPI for the backend, vanilla JS/HTML for the frontend, and the model was trained using PyTorch.

The project creates a canvas to draw on, converts the drawing to a usable 28x28 pixel format, and sends it to the backend for prediction.

The only two routes for the backend are `/` for the home page and `/predict` for prediction API. The prediction route recognizes all English characters and digits. For more details on the architecture, the dataset, and training process, check out [this writeup](https://github.com/intelligent-username/CNN/tree/main/char), which presents the simpler of two models trained for my [writeup on CNNs](https://github.com/intelligent-username/CNN).

## Usage

Once again, the tool is hosted [here](https://ocr.varak.dev) for easy access. 

Draw the character you want to recognize on the left canvas. The right canvas will display the top-k predictions, where k can be adjusted using the slider below it. The slider is capped at 15 since, after 15, all of the predictions are basically guaranteed to be at 0% probability.

To run this project locally, take the following steps:

### Installation

1. Clone this repository.

```bash
git clone https://github.com/intelligent-username/OCR
cd OCR
```

2. Install the Python dependencies to a virtual environment.

```bash
python -m venv OCR-env
OCR-env\Scripts\activate        # On Windows
source OCR-env/bin/activate     # On mac/Linux
pip install -r requirements.txt
```

3. Run the backend.

<!--

Note that, to run the backend from the `backend/` folder, some adjustments to the file paths in `app.py` need to be made, since this version of the project is for the HuggingFace deployment, which uses the root directory as the working directory. The only real difference will be to add `../` to the files paths. Here's the list of changes to make in `app.py`:

Change lines 5 and 6 to:

```python
from utils import predict_image
from model import EMNIST_VGG

```

- Change line 40 to:

```python
model.load_state_dict(t.load("EMNIST_CNN.pth", map_location=device, weights_only=True))

```

- Change line 43 to:

```python
model = t.load("EMNIST_CNN.pth", map_location=device, weights_only=False)
```

- Change line 47 to:

```python
app.mount("/static", StaticFiles(directory="frontend"), name="static")
```

- Change line 51 to:

```python
path = os.path.join("..", "frontend", "index.html")
``` 

You may want to run it from the backend folder if you really want to avoid typing `backend.` at the beginning of the uvicorn command.

-->


```bash
uvicorn backend.app:app --reload
```

4. Once the backend is running, go to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your web browser to access the frontend. This link will appear in the terminal when you run the backend.

## License

This project is licensed under the MIT License. For details, see the [LICENSE](LICENSE) file.
