import pandas as pd
import numpy as np
from dataclasses import dataclass
from scipy import stats

@dataclass
class Metrics:
    ic: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    valid_ratio: float = 0.0


class Evaluator:

    def evaluate(self, signals: pd.DataFrame, close: pd.DataFrame, horizon=5) -> Metrics:

        if signals.empty or close.empty:
            return Metrics()

        try:
            # ===== Forward return =====
            fwd = close.pct_change(horizon).shift(-horizon)

            # tránh look-ahead
            signals = signals.shift(1)

            # align data
            ci = signals.index.intersection(fwd.index)
            cc = signals.columns.intersection(fwd.columns)

            sig = signals.loc[ci, cc]
            ret = fwd.loc[ci, cc]

            ic_list = []
            ls_ret = []

            # ===== tính theo từng ngày (QUAN TRỌNG) =====
            for dt in sig.index:
                s = sig.loc[dt].dropna()
                r = ret.loc[dt].dropna()

                common = s.index.intersection(r.index)
                if len(common) < 10:
                    continue

                s_val = s[common]
                r_val = r[common]

                if s_val.std() < 1e-8 or r_val.std() < 1e-8:
                    continue

                # ===== IC (Spearman chuẩn) =====
                corr, _ = stats.spearmanr(s_val, r_val, nan_policy='omit')
                if not np.isnan(corr):
                    ic_list.append(float(corr))

                # ===== Long-short =====
                rank = s_val.rank(pct=True)

                top = r_val[rank >= 0.8].mean()
                bottom = r_val[rank <= 0.2].mean()

                if not np.isnan(top) and not np.isnan(bottom):
                    ls_ret.append(top - bottom)

            if len(ic_list) < 5:
                return Metrics()

            ic_arr = np.array(ic_list)

            # ===== IC mean =====
            ic_mean = float(np.mean(ic_arr))

            # ===== Sharpe (IC-based) =====
            ic_std = float(np.std(ic_arr)) + 1e-9
            sharpe = ic_mean / ic_std * np.sqrt(252)

            # ===== Win rate =====
            win_rate = float(np.mean(ic_arr > 0))

            # ===== Drawdown (approx) =====
            cum = np.cumsum(ic_arr)
            rolling_max = np.maximum.accumulate(cum)
            dd = float((cum - rolling_max).min())

            return Metrics(
                ic=round(ic_mean, 4),
                sharpe=round(sharpe, 4),
                max_drawdown=round(dd, 4),
                win_rate=round(win_rate, 4),
                valid_ratio=round(len(ic_arr) / len(sig), 4)
            )

        except Exception:
            return Metrics()