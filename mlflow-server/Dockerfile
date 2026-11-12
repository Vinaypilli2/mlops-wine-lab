
FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==2.16.2 psycopg2-binary google-cloud-storage

CMD exec mlflow server --host 0.0.0.0 --port ${PORT:-8080} --backend-store-uri "postgresql+psycopg2://mlflow:${DB_PASS}@/mlflow?host=/cloudsql/${INSTANCE}" --artifacts-destination ${ARTIFACT_URI} --serve-artifacts
