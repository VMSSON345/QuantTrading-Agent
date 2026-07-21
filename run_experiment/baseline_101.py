import json
import numpy as np
import pandas as pd
from src.backtest.backtest_engine import BacktestEngine
from scipy import stats
import logging
from src.data.data_loader import DataLoader
import warnings
from src.data.preprocessor import Preprocessor

logger = logging.getLogger(__name__)

def run_alpha101_baseline(
    alpha_json_path: str,
    data: dict,                # dict giống hệt bạn truyền vào BacktestEngine
    forward_days: int = 5,
) -> pd.DataFrame:
    """
    Chạy toàn bộ 101 alpha trên cùng data và BacktestEngine hiện tại.
    Trả về DataFrame kết quả IC/ICIR để so sánh với QuantAgent.
    """
    with open(alpha_json_path) as f:
        alphas = json.load(f)

    engine = BacktestEngine(data)
    results = []

    for alpha in alphas:
        alpha_id = alpha["id"]
        inputs   = alpha["inputs"]
        code     = alpha["code"]

        try:
            # Lấy đúng các cột data mà alpha cần
            namespace = {}
            for key in inputs:
                if key not in data:
                    raise ValueError(f"Thiếu input '{key}' trong data")
                namespace[key] = data[key].copy()
            namespace["np"] = np
            namespace["pd"] = pd

            # Thực thi code alpha → sinh signal

            code_fixed = code.replace(
            "pct_change()",
            "pct_change(fill_method=None)"
        )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                exec(code_fixed, namespace)
            signal = namespace.get("signal")

            if signal is None:
                raise ValueError("Code không sinh ra biến 'signal'")
            if not isinstance(signal, pd.DataFrame):
                raise ValueError(f"signal phải là DataFrame, nhận {type(signal)}")
            if signal.empty:
                raise ValueError("signal rỗng")

            if signal.isnull().all().all():
                raise ValueError("signal toàn NaN")

            # Gọi evaluate() của BacktestEngine hiện tại
            metrics = engine.evaluate(signal, forward_days=forward_days)

            ic   = metrics.get("ic", 0)
            icir = metrics.get("icir", 0.0)
            tic = metrics.get("tic", 0.0)

            results.append({
                "alpha_id":     alpha_id,
                "inputs":       ", ".join(inputs),
                "ic":           ic,
                "sharpe":       metrics.get("sharpe", 0),
                "icir":         icir,
                "tic":          tic,
                "win_rate":     metrics.get("win_rate", 0),
                "max_drawdown": metrics.get("max_drawdown", 0),
                "n_days":       metrics.get("n_days", 0),
                "valid_ratio":  metrics.get("valid_ratio", 0),
                "status":       "ok",
            })
            logger.info(f"{alpha_id}: IC={ic:.4f}")

        except Exception as e:
            logger.warning(f"{alpha_id} lỗi: {e}")
            results.append({
                "alpha_id": alpha_id,
                "inputs":   ", ".join(inputs),
                "ic":       0.0,
                "status":   f"error: {e}",
            })

    df = pd.DataFrame(results)
    df_ok = df[df["status"] == "ok"].copy()

    # In summary
    print("\n========== Alpha101 Baseline Summary ==========")
    print(f"Tổng alpha:        {len(alphas)}")
    print(f"Chạy thành công:   {len(df_ok)}")
    print(f"Lỗi:               {len(df) - len(df_ok)}")
    print(f"\nIC mean (all ok):  {df_ok['ic'].mean():.4f}")
    print(f"IC std:            {df_ok['ic'].std():.4f}")
    print(f"IC median:         {df_ok['ic'].median():.4f}")
    print(f"IC best:           {df_ok['ic'].max():.4f}  ({df_ok.loc[df_ok['ic'].idxmax(), 'alpha_id']})")
    print(f"IC > 0.02:         {(df_ok['ic'] > 0.02).sum()} alpha")
    print(f"IC > 0:            {(df_ok['ic'] > 0).sum()} alpha")
    print("\nTop 10 alpha:")
    print(df_ok.sort_values("ic", ascending=False).head(10)[
        ["alpha_id", "ic", "sharpe", "win_rate", "n_days"]
    ].to_string(index=False))
    print("================================================\n")

    return df.sort_values("ic", ascending=False)


# ── Chạy ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # data của bạn — giống hệt cách bạn đang load cho QuantAgent
    loader = DataLoader()
    data   = Preprocessor(start="2024-01-01", end="2024-08-31").process(loader.get_data())

    df_results = run_alpha101_baseline(
        alpha_json_path="/workspace/thviet/quant/kb_store/101_alpha.json",
        data=data,
        forward_days=5,
    )

    # Lưu kết quả
    df_results.to_csv("alpha101_baseline_results.csv", index=False)
    print("Đã lưu: alpha101_baseline_results.csv")