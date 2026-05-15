FROM python:3.10

WORKDIR /app
RUN apt-get update && apt-get install -y g++
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000
CMD ["python", "backend/app.py"]