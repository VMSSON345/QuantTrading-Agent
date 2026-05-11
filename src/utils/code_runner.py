import types, traceback, pandas as pd, numpy as np
from typing import Optional


class AlphaBase:
    window_length: int = 20
    inputs: list = None

    def __init__(self):
        if self.inputs is None:
            self.inputs = ["close"]

    def calc(self, data: dict) -> pd.DataFrame:
        raise NotImplementedError


# FIX: thêm các hàm numpy LLM hay dùng như hàm tự do
_BASE_NS = {
    "pd": pd, "np": np,
    "AlphaBase": AlphaBase,
    "DataFrame": pd.DataFrame,
    "Series": pd.Series,
    # Các hàm LLM hay quên prefix np.
    "clip":    np.clip,
    "abs":     np.abs,
    "sqrt":    np.sqrt,
    "log":     np.log,
    "exp":     np.exp,
    "sign":    np.sign,
    "where":   np.where,
    "zeros":   np.zeros,
    "ones":    np.ones,
    "nan":     np.nan,
    "inf":     np.inf,
    "isnan":   np.isnan,
    "isinf":   np.isinf,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "mean":    np.mean,
    "std":     np.std,
    "sum":     np.sum,
    "rank":    pd.DataFrame.rank,
}


class CodeRunner:
    def load_alpha_class(self, code: str):
        namespace = dict(_BASE_NS)
        try:
            exec(compile(code, "<alpha>", "exec"), namespace)
        except Exception as e:
            raise ValueError(f"Lỗi compile code alpha: {e}") from e

        candidates = []
        for obj in namespace.values():
            if (isinstance(obj, type)
                    and hasattr(obj, "calc")
                    and obj.__name__ != "AlphaBase"):
                candidates.append(obj)

        if not candidates:
            raise ValueError("Không tìm thấy class Alpha hợp lệ trong code.")

        return candidates[-1]

    def safe_run(self, code: str, context: Optional[dict] = None) -> dict:
        ns = {**_BASE_NS, **(context or {})}
        try:
            exec(compile(code, "<safe_run>", "exec"), ns)
        except Exception:
            raise RuntimeError(traceback.format_exc())
        return ns