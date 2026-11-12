
import os
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(os.environ["DATA_URI"], sep=";")
X, y = df.iloc[:, :-1], df.iloc[:, -1]

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42
)

params = dict(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

m = xgb.XGBRegressor(
    objective="reg:squarederror", **params
).fit(Xtr, ytr)

pred = m.predict(Xte)

rmse = float(mean_squared_error(yte, pred) ** 0.5)
r2 = float(r2_score(yte, pred))

print("RMSE", rmse, "R2", r2)

mlflow.set_tracking_uri(os.environ["MLFLOW_URI"])
mlflow.set_experiment("wine-quality-training")

with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metrics({"rmse": rmse, "r2": r2})
    mlflow.xgboost.log_model(m, artifact_path="model")

out = os.environ["AIP_MODEL_DIR"].replace("gs://", "/gcs/")
os.makedirs(out, exist_ok=True)

m.get_booster().save_model(os.path.join(out, "model.bst"))

print("saved to", out)
