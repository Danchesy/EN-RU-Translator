FROM python:3.13.0-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
