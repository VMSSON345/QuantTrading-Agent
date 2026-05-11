import pickle
from ..utils.paths import MODEL_DIR


class ModelRegistry:
    def save(self, predictor, name: str):
        MODEL_DIR.mkdir(exist_ok=True)
        payload = {
            "model": predictor.model,
            "feature_names": predictor.feature_names,
            "train_summary": predictor.train_summary,
            "eval_snapshot": predictor.eval_snapshot,
        }
        with open(MODEL_DIR / f"{name}.pkl", "wb") as f:
            pickle.dump(payload, f)

    def load(self, name: str):
        p = MODEL_DIR / f"{name}.pkl"
        if not p.exists():
            return None
        with open(p, "rb") as f:
            return pickle.load(f)