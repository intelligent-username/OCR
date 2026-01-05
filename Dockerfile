FROM python:3.10-slim
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# We run from /code so that 'frontend/' and 'models/' are visible to app.py
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]