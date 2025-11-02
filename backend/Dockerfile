FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && pip install easyocr opencv-python pillow numpy
COPY app.py /app/app.py
COPY ../frontend /app/frontend
COPY ../demo /app/demo
COPY ../data /app/data
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
