
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir "numpy<2" pandas==2.2.2 scikit-learn==1.4.2 xgboost==1.7.6 gcsfs mlflow==2.16.2

COPY . .

CMD ["python", "train.py"]
