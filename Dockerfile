# Python 3.10 (slim) as base
FROM python:3.10-slim

# THhis is the working directory of the container
WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the container
COPY . .

# Run the FastAPI application using uvicorn on all interfaces at port 7860
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]