FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

RUN python /app/scripts/gen_protos.py

ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["python", "-u", "launch.py"]


