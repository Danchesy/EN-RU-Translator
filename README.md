docker build -t translator:latest .
docker run -d --name translator-container -p 8501:8501 translator:latest