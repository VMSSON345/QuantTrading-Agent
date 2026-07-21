import importlib.util
import inspect
import os
import re
import numpy as np
import pandas as pd
from src.backtest.backtest_engine import BacktestEngine
import logging
from src.data.data_loader import DataLoader
from src.data.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


# ── Base class ─────────────────────────────────────────────────────────────────
class AlphaBase:
    """Base class tất cả alpha phải kế thừa."""
    window_length: int = 1
    inputs: list = []

    def calc(self, data: dict) -> pd.DataFrame:
        raise NotImplementedError("Subclass must implement calc()")


# ── Normalize ──────────────────────────────────────────────────────────────────
def normalize_code_str(code_str: str) -> str:
    """
    Chuẩn hóa code string từ LLM output trước khi exec:
    1. Unescape \\n literal → newline thật
    2. Unescape \\\" → "
    3. Xóa block 'class AlphaBase...' LLM tự định nghĩa lại
    4. Sửa 'import pandas as pd, numpy as np' → 2 dòng riêng
    """
    # 1. Unescape \n literal nếu chưa có newline thật
    if "\\n" in code_str and "\n" not in code_str:
        code_str = code_str.replace("\\n", "\n").replace("\\t", "\t")

    # 2. Unescape escaped quotes
    code_str = code_str.replace('\\"', '"')

    # 3. Xóa block 'class AlphaBase' mà LLM tự định nghĩa lại
    code_str = re.sub(
        r"^class AlphaBase\b.*?(?=^class |\Z)",
        "",
        code_str,
        flags=re.MULTILINE | re.DOTALL,
    )

    # 4. Sửa import một dòng không hợp lệ
    code_str = code_str.replace(
        "import pandas as pd, numpy as np",
        "import pandas as pd\nimport numpy as np",
    )

    return code_str.strip()


# ── Loaders ────────────────────────────────────────────────────────────────────
def load_alphas_from_code(code_str: str, alpha_id: str | None = None) -> list[tuple[str, AlphaBase]]:
    """Exec một code string, trả về list (display_name, instance)."""
    namespace = {"AlphaBase": AlphaBase, "pd": pd, "np": np}
    exec(normalize_code_str(code_str), namespace)

    alphas = []
    for name, obj in namespace.items():
        if inspect.isclass(obj) and issubclass(obj, AlphaBase) and obj is not AlphaBase:
            display_name = f"{alpha_id}:{name}" if alpha_id else name
            alphas.append((display_name, obj()))
    return alphas


def load_alphas_from_path(path: str) -> list[tuple[str, AlphaBase]]:
    """Load từ file .py hoặc thư mục."""
    alphas = []

    def _from_file(filepath: str):
        module_name = os.path.splitext(os.path.basename(filepath))[0]
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        module.AlphaBase = AlphaBase
        module.pd = pd
        module.np = np
        spec.loader.exec_module(module)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, AlphaBase) and obj is not AlphaBase and obj.__module__ == module.__name__:
                alphas.append((name, obj()))

    if os.path.isfile(path):
        _from_file(path)
    elif os.path.isdir(path):
        for fname in sorted(os.listdir(path)):
            if fname.endswith(".py") and not fname.startswith("_"):
                _from_file(os.path.join(path, fname))
    else:
        raise ValueError(f"Path không tồn tại: {path}")
    return alphas


def load_alphas_from_code_list(code_list: list) -> list[tuple[str, AlphaBase]]:
    """
    Load từ list. Mỗi phần tử có thể là:
    - str  : code string (newline thật hoặc \\n literal)
    - dict : {"id": "...", "code": "..."}
    - tuple: ("id", "code string")
    """
    alphas = []
    for i, item in enumerate(code_list):
        if isinstance(item, str):
            alpha_id, code_str = f"alpha_{i:03d}", item
        elif isinstance(item, dict):
            alpha_id, code_str = item.get("id", f"alpha_{i:03d}"), item["code"]
        elif isinstance(item, tuple) and len(item) == 2:
            alpha_id, code_str = item
        else:
            logger.warning(f"Bỏ qua index {i}: format không hợp lệ ({type(item)})")
            continue

        try:
            loaded = load_alphas_from_code(code_str, alpha_id=alpha_id)
            if not loaded:
                logger.warning(f"{alpha_id}: không tìm thấy class AlphaBase nào")
            alphas.extend(loaded)
        except Exception as e:
            logger.warning(f"{alpha_id}: lỗi khi load — {e}")

    return alphas


# ── Main runner ────────────────────────────────────────────────────────────────
def run_alpha_baseline(
    alpha_source,
    data: dict,
    forward_days: int = 5,
) -> pd.DataFrame:
    """
    alpha_source nhận:
    - str path file/thư mục .py
    - str code (single alpha)
    - list[str | dict | tuple]
    - list[tuple[str, AlphaBase]]  (đã load sẵn)
    """
    if isinstance(alpha_source, list):
        if alpha_source and isinstance(alpha_source[0], tuple) and isinstance(alpha_source[0][1], AlphaBase):
            alpha_list = alpha_source
        else:
            alpha_list = load_alphas_from_code_list(alpha_source)
    elif isinstance(alpha_source, str):
        if os.path.exists(alpha_source):
            alpha_list = load_alphas_from_path(alpha_source)
        else:
            alpha_list = load_alphas_from_code(alpha_source)
    else:
        raise TypeError(f"alpha_source không hợp lệ: {type(alpha_source)}")

    if not alpha_list:
        raise ValueError("Không tìm thấy alpha class nào.")

    engine  = BacktestEngine(data)
    results = []

    for alpha_name, alpha_instance in alpha_list:
        try:
            missing = [k for k in alpha_instance.inputs if k not in data]
            if missing:
                raise ValueError(f"Thiếu input: {missing}")

            alpha_data = {k: data[k].copy() for k in alpha_instance.inputs}
            signal     = alpha_instance.calc(alpha_data)

            if signal is None:
                raise ValueError("calc() trả về None")
            if not isinstance(signal, pd.DataFrame):
                raise ValueError(f"calc() phải trả về DataFrame, nhận {type(signal)}")
            if signal.empty:
                raise ValueError("signal rỗng")
            if signal.isnull().all().all():
                raise ValueError("signal toàn NaN")

            metrics = engine.evaluate(signal, forward_days=forward_days)
            ic      = metrics.get("ic",   0)
            icir    = metrics.get("icir", 0.0)
            tic     = metrics.get("tic",  0.0)

            results.append({
                "alpha_id":     alpha_name,
                "inputs":       ", ".join(alpha_instance.inputs),
                "window":       alpha_instance.window_length,
                "ic":           ic,
                "sharpe":       metrics.get("sharpe",       0),
                "icir":         icir,
                "tic":          tic,
                "win_rate":     metrics.get("win_rate",     0),
                "max_drawdown": metrics.get("max_drawdown", 0),
                "n_days":       metrics.get("n_days",       0),
                "valid_ratio":  metrics.get("valid_ratio",  0),
                "status":       "ok",
            })
            logger.info(f"{alpha_name}: IC={ic:.4f}")

        except Exception as e:
            logger.warning(f"{alpha_name} lỗi: {e}")
            results.append({
                "alpha_id": alpha_name,
                "inputs":   ", ".join(getattr(alpha_instance, "inputs", [])),
                "ic":       0.0,
                "status":   f"error: {e}",
            })

    df    = pd.DataFrame(results)
    df_ok = df[df["status"] == "ok"].copy()

    print("\n========== Alpha Baseline Summary ==========")
    print(f"Tổng alpha:       {len(alpha_list)}")
    print(f"Thành công:       {len(df_ok)}")
    print(f"Lỗi:              {len(df) - len(df_ok)}")
    if not df_ok.empty:
        best = df_ok.loc[df_ok["ic"].idxmax(), "alpha_id"]
        print(f"\nIC mean:          {df_ok['ic'].mean():.4f}")
        print(f"IC std:           {df_ok['ic'].std():.4f}")
        print(f"IC median:        {df_ok['ic'].median():.4f}")
        print(f"IC best:          {df_ok['ic'].max():.4f}  ({best})")
        print(f"IC > 0.02:        {(df_ok['ic'] > 0.02).sum()}")
        print(f"IC > 0:           {(df_ok['ic'] > 0).sum()}")
        print("\nTop 10:")
        print(df_ok.sort_values("ic", ascending=False).head(10)[
            ["alpha_id", "ic", "sharpe", "win_rate", "icir", "tic", "n_days"]
        ].to_string(index=False))
    print("=============================================\n")

    return df.sort_values("ic", ascending=False)


# ── Alpha list ─────────────────────────────────────────────────────────────────
ALPHA_CODES = [
    ("accumulation_recovery", """
import pandas as pd, numpy as np

class AlphaBase:
    window_length=20; inputs=["close","volume"]
    def calc(self,data): pass

class AccumulationRecoveryAlpha(AlphaBase):
    def __init__(self):
        self.window_length=20
        self.inputs=["close","volume"]

    def calc(self, data:dict)->pd.DataFrame:
        close  = data["close"].ffill().fillna(0)
        volume = data["volume"].ffill().fillna(1)

        cond1 = close < close.shift(1)
        cond2 = close.shift(1) < close.shift(2)
        cond3 = close.shift(2) < close.shift(3)
        condition = cond1 & cond2 & cond3

        delta_volume = volume / volume.shift(1) - 1
        volume_mean = volume.rolling(window=10).mean()
        delta_volume_normalized = (delta_volume / volume_mean).clip(-2, 2)

        signal = delta_volume_normalized * condition.astype(int)
        signal = signal.rank(axis=1, pct=True) - 0.5

        return signal.reindex(index=close.index, columns=close.columns).fillna(0)
"""),

    ("ma_distance_volume", """
import pandas as pd, numpy as np

class AlphaBase:
    window_length=20; inputs=["close"]
    def calc(self,data): pass

class MyAlpha(AlphaBase):
    def __init__(self):
        self.window_length=20
        self.inputs=["close","volume"]

    def calc(self,data:dict)->pd.DataFrame:
        close  = data["close"].ffill().fillna(0)
        volume = data["volume"].ffill().fillna(1)

        ma_20  = close.rolling(self.window_length).mean()
        std_20 = close.rolling(self.window_length).std()
        distance = -(close - ma_20) / std_20

        volume_filter = (volume / volume.rolling(10).mean()).clip(0,2)
        signal = distance * volume_filter
        signal = signal.rank(axis=1, pct=True) - 0.5

        return signal.reindex(index=close.index, columns=close.columns).fillna(0)
"""),

    ("volume_accumulation", """
import pandas as pd, numpy as np

class AlphaBase:
    window_length=20; inputs=["close", "high", "low", "volume"]
    def calc(self,data): pass

class VolumeAccumulationAlpha(AlphaBase):
    def __init__(self):
        self.window_length=20
        self.inputs=["close", "high", "low", "volume"]

    def calc(self,data:dict)->pd.DataFrame:
        close  = data["close"].ffill().fillna(0)
        high   = data["high"].ffill().fillna(0)
        low    = data["low"].ffill().fillna(0)
        volume = data["volume"].ffill().fillna(1)

        avg_volume_10d = volume.rolling(10).mean()
        price_range = (high - low) / close
        signal = (volume / avg_volume_10d) * (1 - price_range)
        signal = signal.rank(axis=1, pct=True) - 0.5

        return signal
"""),

    ("consecutive_down", """
import pandas as pd, numpy as np

class AlphaBase:
    window_length=20; inputs=["close","volume"]
    def calc(self,data): pass

class ConsecutiveDownAlpha(AlphaBase):
    def __init__(self):
        self.window_length=20
        self.inputs=["close","volume"]

    def calc(self, data:dict)->pd.DataFrame:
        close = data["close"].ffill().fillna(0)

        delta_close = close.diff().fillna(0)
        delta_close = delta_close.mask(delta_close > 0, 0)
        sum_down = delta_close.rolling(5).sum().abs().fillna(0)

        signal = sum_down / 5
        signal = signal.rank(axis=1, pct=True) - 0.5

        return signal.reindex(index=close.index, columns=close.columns).fillna(0)
"""),

    ("volume_price_accumulation", """
import pandas as pd, numpy as np

class AlphaBase:
    window_length=20; inputs=["close","volume","high","low"]
    def calc(self,data): pass

class AccumulationAlpha(AlphaBase):
    def __init__(self):
        self.window_length=20
        self.inputs=["close","volume","high","low"]

    def calc(self,data:dict)->pd.DataFrame:
        close  = data["close"].ffill().fillna(0)
        volume = data["volume"].ffill().fillna(1)
        high   = data["high"].ffill().fillna(0)
        low    = data["low"].ffill().fillna(0)

        volume_ma = volume.rolling(window=self.window_length).mean()
        epsilon = 1e-6
        signal = ((volume - volume_ma) / volume_ma) * (1 / (high - low + epsilon))
        signal = signal.rank(axis=1, pct=True) - 0.5

        return signal.reindex(index=close.index, columns=close.columns).fillna(0)
"""),

    ("momentum_loss", """
import pandas as pd, numpy as np

class AlphaBase:
    window_length=20; inputs=["close"]
    def calc(self,data): pass

class MyAlpha(AlphaBase):
    def __init__(self):
        self.window_length=20
        self.inputs=["close"]

    def calc(self,data:dict)->pd.DataFrame:
        close = data["close"].ffill().fillna(0)

        delta_close = close.pct_change()
        indicator = ((delta_close < delta_close.shift(1)) & (delta_close.shift(1) > 0)) * 1.0
        momentum_loss = delta_close * indicator
        sum_momentum_loss = momentum_loss.rolling(5).sum()

        signal = -sum_momentum_loss.rank(axis=1, pct=True) + 0.5

        return signal
"""),

    ("bb_upper_breakout", """
import pandas as pd, numpy as np

class MyAlpha(AlphaBase):
    def __init__(self):
        self.window_length=20
        self.inputs=["close","volume"]

    def calc(self,data:dict)->pd.DataFrame:
        close = data["close"].ffill().fillna(0)
        ma  = close.rolling(20).mean()
        std = close.rolling(20).std()

        upper = ma + 2 * std
        signal = -(close - upper).clip(lower=0) / std
        signal = signal.rank(axis=1, pct=True) - 0.5

        return signal
"""),

    ("bb_lower_ewma", """
import pandas as pd, numpy as np

class MyAlpha(AlphaBase):
    def __init__(self):
        self.window_length=20
        self.inputs=["close","volume"]

    def calc(self,data:dict)->pd.DataFrame:
        close  = data["close"].ffill().fillna(0)
        volume = data["volume"].ffill().fillna(1)

        ma  = close.ewm(span=20, adjust=False).mean()
        std = close.ewm(span=20, adjust=False).std()

        volume_filter = (volume / volume.rolling(10).mean()).clip(0,2)
        signal = -((close - ma) / (2 * std)) * volume_filter
        signal = signal.rank(axis=1, pct=True) - 0.5

        return signal.reindex(index=close.index, columns=close.columns).fillna(0)
"""),

    ("stochastic_volume", """
import pandas as pd, numpy as np

class AlphaBase:
    window_length=20; inputs=["close"]
    def calc(self,data): pass

class MyAlpha(AlphaBase):
    def __init__(self):
        self.window_length=20
        self.inputs=["close","high","low","volume"]

    def calc(self,data:dict)->pd.DataFrame:
        close  = data["close"].ffill().fillna(0)
        high   = data["high"].ffill().fillna(0)
        low    = data["low"].ffill().fillna(0)
        volume = data["volume"].ffill().fillna(1)

        signal = (close - low) / (high - low).replace(0, np.nan)
        signal = -signal

        volume_filter = (volume / volume.rolling(10).mean()).clip(0,2)
        signal = signal * volume_filter
        signal = signal.rank(axis=1, pct=True) - 0.5

        return signal.reindex(index=close.index, columns=close.columns).fillna(0)
"""),
]


# ── Chạy ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    loader = DataLoader()
    data   = Preprocessor().process(loader.get_data())

    df_results = run_alpha_baseline(
        alpha_source=ALPHA_CODES,
        data=data,
        forward_days=5,
    )

    df_results.to_csv("alpha_baseline_results.csv", index=False)
    print("Đã lưu: alpha_baseline_results.csv")