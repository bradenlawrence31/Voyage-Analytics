FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ ./models/
COPY api/app.py ./api/app.py
COPY datasets/ ./datasets/

EXPOSE 5000

CMD ["python", "api/app.py"]
