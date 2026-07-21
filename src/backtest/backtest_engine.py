import pandas as pd
import numpy as np
from loguru import logger
from scipy import stats


class BacktestEngine:
    def __init__(self, data: dict):
        assert isinstance(data, dict), f"data phai la dict, nhan {type(data)}"
        self.data = data

    def compute_volume_ratio(self, window: int = 20) -> pd.DataFrame:
        volume = self.data.get("volume", None)

        if volume is None or volume.empty:
            logger.warning("compute_volume_ratio: thiếu volume")
            return pd.DataFrame()

        volume = volume.ffill().fillna(0)
        avg_past = volume.shift(1).rolling(window, min_periods=window).mean()
        ratio = volume / avg_past
        return ratio.replace([np.inf, -np.inf], np.nan)

    def compute_signals(self, alpha_cls) -> pd.DataFrame:
        try:
            instance = alpha_cls()
            needed = getattr(instance, 'inputs', None) or ["close"]
            if not isinstance(needed, list):
                needed = list(needed)

            close = self.data.get("close", pd.DataFrame())
            if close.empty:
                return pd.DataFrame()

            data_for_alpha = {}
            for key in needed:
                if key in self.data:
                    data_for_alpha[key] = self.data[key].copy()
                elif key == "volume_ratio":
                    data_for_alpha[key] = self.compute_volume_ratio()
                else:
                    logger.warning(f"compute_signals: thiếu key '{key}', dùng close thay thế")
                    data_for_alpha[key] = pd.DataFrame(0, index=close.index, columns=close.columns)

            try:
                raw = instance.calc(data_for_alpha)
            except AttributeError as e:
                logger.warning(f"compute_signals: calc() lỗi AttributeError ({e}), thử wrap ndarray")
                return pd.DataFrame()
            except Exception as e:
                logger.warning(f"compute_signals: calc() lỗi ({e})")
                return pd.DataFrame()

            if raw is None:
                logger.warning("compute_signals: calc() trả về None")
                return pd.DataFrame()

            if isinstance(raw, np.ndarray):
                if raw.ndim == 1:
                    raw = pd.DataFrame(
                        np.outer(raw, np.ones(len(close.columns))),
                        index=close.index[:len(raw)],
                        columns=close.columns,
                    )
                else:
                    rows = min(raw.shape[0], len(close.index))
                    cols = min(raw.shape[1], len(close.columns))
                    raw = pd.DataFrame(
                        raw[:rows, :cols],
                        index=close.index[:rows],
                        columns=close.columns[:cols],
                    )
            elif isinstance(raw, pd.Series):
                raw = pd.DataFrame(
                    np.outer(raw.values, np.ones(len(close.columns))),
                    index=raw.index,
                    columns=close.columns,
                )
            elif isinstance(raw, (int, float, np.floating, np.integer)):
                logger.warning("compute_signals: calc() trả về scalar, bỏ qua")
                return pd.DataFrame()
            elif not isinstance(raw, pd.DataFrame):
                logger.warning(f"compute_signals: calc() trả về {type(raw)}, bỏ qua")
                return pd.DataFrame()

            if raw.shape != close.shape:
                raw = raw.reindex(index=close.index, columns=close.columns).fillna(0)

            try:
                vol_ratio = self.compute_volume_ratio()
                if not vol_ratio.empty:
                    vol_ratio = vol_ratio.reindex_like(raw)
                    #liquidity_mask = (vol_ratio > 1.0).astype(float)
                    #raw = raw * (0.5 + 0.5 * liquidity_mask)
                    raw = raw.where(vol_ratio > 1.0, other=np.nan)
            except Exception as e:
                logger.warning(f"liquidity filter lỗi: {e}")

            return raw.replace([np.inf, -np.inf], np.nan)

        except Exception as e:
            logger.error(f"compute_signals lỗi: {e}")
            return pd.DataFrame()

    def evaluate(self, signal: pd.DataFrame, forward_days: int = 5) -> dict:
        try:
            close = self.data.get("close", pd.DataFrame())
            if close.empty or signal.empty:
                return self._zero_metrics()

            close = close.copy()
            signal = signal.copy()
            close.index = pd.to_datetime(close.index)
            signal.index = pd.to_datetime(signal.index)
            close = close.ffill().bfill()

            valid_cols = close.columns[close.std() > 0.01]
            close = close[valid_cols]
            signal = signal.reindex(columns=valid_cols)#.fillna(0)

            fwd_ret = close.pct_change(forward_days).shift(-forward_days)
            fwd_ret = fwd_ret.replace([np.inf, -np.inf], np.nan)

            common_idx = signal.index.intersection(fwd_ret.index)
            sig = signal.loc[common_idx] #shift(1)ogiua
            if sig.abs().max().max() > 1:
                sig = sig.rank(axis=1, pct=True) - 0.5
            ret = fwd_ret.loc[common_idx]

            cutoff = sig.index[-forward_days] if len(sig) > forward_days else sig.index[-1]
            sig = sig.loc[sig.index < cutoff]
            ret = ret.loc[ret.index < cutoff]

            if len(sig) < 20:
                return self._zero_metrics()

            ic_series = []
            #moi
            ic_dates = []
            top_bucket_returns = []
            bottom_bucket_returns = []
            long_short_spreads = []

            for dt in sig.index:
                s = sig.loc[dt].dropna()
                r = ret.loc[dt].dropna()
                common = s.index.intersection(r.index)
                if len(common) < 5:
                    continue
                s_val = s[common].values
                r_val = r[common].values
                if s_val.std() < 1e-8 or r_val.std() < 1e-8:
                    continue
                try:
                    corr, _ = stats.spearmanr(s_val, r_val, nan_policy='omit')
                    if not np.isnan(corr):
                        ic_dates.append(dt)
                        ic_series.append(float(corr))
                except Exception:
                    continue

                try:
                    rank_pct = s[common].rank(method="average", pct=True)
                    top_mask = rank_pct >= 0.8
                    bottom_mask = rank_pct <= 0.2
                    if top_mask.any() and bottom_mask.any():
                        top_ret = float(r[common][top_mask].mean())
                        bottom_ret = float(r[common][bottom_mask].mean())
                        top_bucket_returns.append(top_ret)
                        bottom_bucket_returns.append(bottom_ret)
                        long_short_spreads.append(top_ret - bottom_ret)
                except Exception:
                    continue

            if len(ic_series) < 5:
                return self._zero_metrics()
            
            ic_ts = pd.Series(ic_series, index=pd.to_datetime(ic_dates))
            ic_arr = ic_ts.values
            yearly_ic = ic_ts.groupby(ic_ts.index.year).mean()

            ic_arr = np.array(ic_series)
            ic_mean = float(np.mean(ic_arr))
            ic_std = float(np.std(ic_arr)) + 1e-8

            #moi
            n_days_ic = len(ic_series)
            
            # 1. ICIR (Information Coefficient Information Ratio)
            icir = float(ic_mean / ic_std)
            
            # 2. tIC (t-statistic) = ICIR * sqrt(N)
            t_ic = float(icir * np.sqrt(n_days_ic))
            
            # 3. 95% Confidence Intervals (CI)
            margin_of_error = 1.96 * (ic_std / np.sqrt(n_days_ic))
            ci_lower = round(float(ic_mean - margin_of_error), 6)
            ci_upper = round(float(ic_mean + margin_of_error), 6)

            portfolio_returns = np.array(long_short_spreads)

            if len(portfolio_returns) > 1:
                mean_ret = np.mean(portfolio_returns)
                std_ret = np.std(portfolio_returns)

                if std_ret > 1e-8:
                    annual_factor = np.sqrt(252 / forward_days)

                    sharpe = mean_ret / std_ret * annual_factor
                else:
                    sharpe = 0.0
            else:
                sharpe = 0.0
            #winrate 
            win_rate = float(np.mean(ic_arr > 0))
            
            #maxdrawdown
            equity = np.cumprod(1 + portfolio_returns)

            rolling_max = np.maximum.accumulate(equity)

            drawdown = (equity - rolling_max) / rolling_max

            max_dd = float(drawdown.min()) if len(drawdown) else 0.0



            vol_ratio = self.compute_volume_ratio()
            if not vol_ratio.empty:
                vol_ratio = vol_ratio.reindex_like(signal)
                avg_liquidity = float(np.nanmean(vol_ratio.values))
                high_liq_ratio = float(np.mean(vol_ratio.values > 1.2))
            else:
                avg_liquidity = 0.0
                high_liq_ratio = 0.0

            top_bucket_return = float(np.mean(top_bucket_returns)) if top_bucket_returns else 0.0
            bottom_bucket_return = float(np.mean(bottom_bucket_returns)) if bottom_bucket_returns else 0.0
            long_short_spread = float(np.mean(long_short_spreads)) if long_short_spreads else 0.0

            return {
                "ic": round(ic_mean, 6),
                "sharpe": round(sharpe, 6),
                "max_drawdown": round(max_dd, 6),
                "win_rate": round(win_rate, 4),
                "valid_ratio": round(len(ic_series) / max(len(sig), 1), 4),
                "n_days": len(ic_series),
                "avg_liquidity": round(avg_liquidity, 4),
                "high_liq_ratio": round(high_liq_ratio, 4),
                "top_bucket_return": round(top_bucket_return, 6),
                "bottom_bucket_return": round(bottom_bucket_return, 6),
                "long_short_spread": round(long_short_spread, 6),
                "icir": round(icir, 6),
                "tic": round(t_ic, 4),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "ic_ts": {str(year): round(float(val), 6) for year, val in yearly_ic.items()},
            }

        except Exception as e:
            logger.error(f"evaluate lỗi: {e}")
            return self._zero_metrics()

    @staticmethod
    def _zero_metrics() -> dict:
        return {
            "ic": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "valid_ratio": 0.0,
            "n_days": 0,
            "avg_liquidity": 0.0,
            "high_liq_ratio": 0.0,
            "icir": 0.0,
            "tic": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "ic_ts": {},
        }
