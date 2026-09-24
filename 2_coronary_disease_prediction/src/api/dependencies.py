from src.ml.libs.session import MLSession
from src.ml.models.xgb import XGBoost

config = {
    "mode": "clf_binary",
    "raw_model": XGBoost
}

session = MLSession(
    config=config,
)

session.load(model_path="./src/api/models/xgboost.joblib")

def get_session() -> MLSession:
    return session
