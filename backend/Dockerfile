FROM python:3.11-slim

# root workspace
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy into /app/backend
COPY . /app/backend

# move working dir to backend (optional but clean)
WORKDIR /app/backend

# IMPORTANT: your module path is backend.main
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
