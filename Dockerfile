FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN python /app/scripts/gen_protos.py

ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["python", "-u", "-m", "app.main"]


