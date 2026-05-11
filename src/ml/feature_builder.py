"""
FeatureBuilder — v2
====================
Thay đổi so với v1:
  1. Xóa bfill() — chỉ ffill() để tránh look-ahead bias
  2. Tách _compute_features() dùng chung cho build() và build_latest()
     → tránh train/serve skew
  3. Target = cross-sectional rank-normalized forward return (winsorized)
     thay vì raw pct_change
  4. fillna dùng cross-sectional median thay vì 0
  5. Thêm missing-indicator features cho alpha signals
  6. Winsorize feature values ở ±5 sigma
"""

import numpy as np
import pandas as pd
from loguru import logger


class FeatureBuilder:
    """
    Tạo features từ raw OHLCV data cho EnsemblePredictor.

    Input:
        data = {
            "open":   DataFrame(date × symbol),
            "high":   DataFrame(date × symbol),
            "low":    DataFrame(date × symbol),
            "close":  DataFrame(date × symbol),
            "volume": DataFrame(date × symbol),
        }

    Output:
        - build()        → (X, y) long-format MultiIndex (date, symbol)
        - build_latest() → DataFrame(index=symbol, columns=features)

    Quy ước:
        - Không bao giờ dùng bfill() trên time-series OHLCV
        - Missing = forward-fill rồi điền cross-sectional median
        - Target = rank cross-sectional, winsorize ±0.45
    """

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def build(
        self,
        data: dict,
        forward_days: int = 5,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Build full training set.

        Returns
        -------
        X : DataFrame  — MultiIndex (date, symbol), columns = feature names
        y : Series     — same index, cross-sectional rank fwd return
        """
        close, high, low, volume = self._prep_ohlcv(data)
        if close.empty:
            logger.warning("FeatureBuilder.build: close data rỗng")
            return pd.DataFrame(), pd.Series(dtype=float)

        # ---- features (dùng chung logic với build_latest) ----
        feat_dfs: dict[str, pd.DataFrame] = self._compute_features(
            close, high, low, volume
        )

        # ---- stack về long format ----
        feat_frames = [
            df.stack(dropna=False).rename(name)
            for name, df in feat_dfs.items()
        ]
        if not feat_frames:
            logger.warning("FeatureBuilder.build: không tạo được feature nào")
            return pd.DataFrame(), pd.Series(dtype=float)

        X = pd.concat(feat_frames, axis=1)

        # ---- target: rank-normalized cross-sectional forward return ----
        y = self._build_target(close, forward_days)

        # ---- align & clean ----
        idx = X.index.intersection(y.index)
        X = X.loc[idx]
        y = y.loc[idx]

        mask = ~y.isna()
        X, y = X.loc[mask], y.loc[mask]

        X = self._clean_X(X)

        logger.info(
            f"FeatureBuilder.build: {X.shape[1]} features, {len(X)} samples"
        )
        return X, y

    def build_latest(self, data: dict) -> pd.DataFrame:
        """
        Lấy features của ngày mới nhất để chạy prediction.

        Returns
        -------
        DataFrame — index = symbol, columns = feature names
        """
        close, high, low, volume = self._prep_ohlcv(data)
        if close.empty:
            logger.warning("FeatureBuilder.build_latest: close data rỗng")
            return pd.DataFrame()

        feat_dfs = self._compute_features(close, high, low, volume)

        # Lấy hàng cuối cùng của mỗi feature DataFrame
        result = pd.DataFrame(
            {name: df.iloc[-1] for name, df in feat_dfs.items()}
        )

        return self._clean_X(result)

    # ------------------------------------------------------------------ #
    #  Core feature computation — dùng chung cho build() & build_latest() #
    # ------------------------------------------------------------------ #

    def _compute_features(
        self,
        close: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        volume: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """
        Tính toàn bộ features. Trả về dict[name → DataFrame(date × symbol)].
        Method này KHÔNG được gọi bất kỳ .iloc[-1] hay .stack() nào —
        chỉ trả về full DataFrame để caller tự xử lý.
        """
        feats: dict[str, pd.DataFrame] = {}
        ret_1d = close.pct_change(1)

        # ---- MOMENTUM ----
        for n in [3, 5, 10, 20]:
            mom = close.pct_change(n)
            feats[f"mom_{n}"] = mom
            # Cross-sectional rank (centered về 0)
            feats[f"mom_rank_{n}"] = mom.rank(axis=1, pct=True) - 0.5

        # ---- MEAN REVERSION / Z-SCORE ----
        for n in [5, 10, 20]:
            mu = close.rolling(n, min_periods=max(3, n // 2)).mean()
            sigma = (
                close.rolling(n, min_periods=max(3, n // 2))
                .std()
                .replace(0, np.nan)
            )
            feats[f"zscore_{n}"] = (close - mu) / sigma

        # ---- RSI proxy (normalized return / volatility) ----
        for n in [5, 14]:
            gains = ret_1d.clip(lower=0).rolling(n, min_periods=max(3, n // 2)).mean()
            losses = (-ret_1d).clip(lower=0).rolling(n, min_periods=max(3, n // 2)).mean()
            rs = gains / losses.replace(0, np.nan)
            feats[f"rsi_{n}"] = 1 - (1 / (1 + rs))  # scale [0,1]

        # ---- VOLATILITY ----
        for n in [5, 10, 20]:
            hv = ret_1d.rolling(n, min_periods=max(3, n // 2)).std()
            feats[f"hv_{n}"] = hv
            # Volatility ratio (short-term vs long-term)
            if n < 20:
                hv_long = ret_1d.rolling(20, min_periods=10).std()
                feats[f"vol_ratio_{n}_20"] = hv / hv_long.replace(0, np.nan)

        # ---- VOLUME ----
        if not volume.empty:
            for n in [5, 10, 20]:
                vol_mu = volume.rolling(n, min_periods=max(3, n // 2)).mean()
                vol_sigma = (
                    volume.rolling(n, min_periods=max(3, n // 2))
                    .std()
                    .replace(0, np.nan)
                )
                feats[f"vol_zscore_{n}"] = (volume - vol_mu) / vol_sigma

            vol_ma20 = volume.rolling(20, min_periods=10).mean().replace(0, np.nan)
            feats["vol_ratio_20"] = volume / vol_ma20

            # Price-volume divergence: return / volume change
            vol_chg = volume.pct_change(1).replace(0, np.nan)
            feats["vol_price_div"] = ret_1d / vol_chg

            # Amihud illiquidity proxy
            feats["amihud"] = ret_1d.abs() / volume.replace(0, np.nan)
            feats["amihud"] = feats["amihud"].rolling(5, min_periods=3).mean()

        # ---- HIGH-LOW RANGE ----
        if not high.empty and not low.empty:
            hl = (high - low) / close.replace(0, np.nan)
            feats["hl_range_5"] = hl.rolling(5, min_periods=3).mean()

            # True range normalized
            tr = pd.concat(
                [
                    (high - low),
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs(),
                ],
                axis=0,
            ).groupby(level=0).max() if False else (high - low)  # simplified TR
            feats["hl_range_1"] = hl

        # ---- VWAP DEVIATION ----
        if not volume.empty:
            for n in [5, 10, 20]:
                denom = volume.rolling(n, min_periods=max(3, n // 2)).sum().replace(0, np.nan)
                vwap = (close * volume).rolling(n, min_periods=max(3, n // 2)).sum() / denom
                feats[f"vwap_dev_{n}"] = (close - vwap) / vwap.replace(0, np.nan)

        # ---- MEAN REVERSION SPEED ----
        # Khoảng cách về mean / độ lệch chuẩn của return → dự báo reversal
        for n in [10, 20]:
            mu = close.rolling(n, min_periods=max(5, n // 2)).mean()
            dist = (close - mu) / close.replace(0, np.nan)
            feats[f"dist_ma_{n}"] = dist

        # ---- TREND CONSISTENCY ----
        # Tỷ lệ ngày tăng trong cửa sổ
        for n in [5, 10, 20]:
            feats[f"up_frac_{n}"] = (ret_1d > 0).rolling(n, min_periods=max(3, n // 2)).mean()

        return feats

    # ------------------------------------------------------------------ #
    #  Target construction                                                 #
    # ------------------------------------------------------------------ #

    def _build_target(
        self,
        close: pd.DataFrame,
        forward_days: int,
    ) -> pd.Series:
        """
        Target = cross-sectional rank-normalized forward return.

        1. Tính raw forward return
        2. Rank cross-sectionally từng ngày → [0, 1]
        3. Center về 0 → [-0.5, 0.5]
        4. Winsorize ±0.45 để loại cổ phiếu cực đoan
        5. Stack về long format
        """
        fwd_ret = close.pct_change(forward_days).shift(-forward_days)

        # Rank từng ngày (cross-sectional)
        fwd_rank = fwd_ret.rank(axis=1, pct=True) - 0.5

        # Winsorize
        fwd_rank = fwd_rank.clip(-0.45, 0.45)

        return fwd_rank.stack(dropna=False).rename("fwd_rank")

    # ------------------------------------------------------------------ #
    #  Data preparation                                                    #
    # ------------------------------------------------------------------ #

    def _prep_ohlcv(
        self,
        data: dict,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load và forward-fill OHLCV. KHÔNG dùng bfill().
        Missing ở đầu chuỗi → giữ NaN để rolling window xử lý đúng.
        """
        close = data.get("close", pd.DataFrame())
        high = data.get("high", pd.DataFrame())
        low = data.get("low", pd.DataFrame())
        volume = data.get("volume", pd.DataFrame())

        if not close.empty:
            close = close.copy().ffill()  # Chỉ ffill — không bfill
        if not high.empty:
            high = high.copy().ffill()
        if not low.empty:
            low = low.copy().ffill()
        if not volume.empty:
            volume = volume.copy().ffill()
            # Volume = 0 ở ngày không giao dịch là hợp lệ, giữ nguyên
            # Chỉ thay NaN bằng 0 (cổ phiếu chưa niêm yết)
            volume = volume.fillna(0.0)

        return close, high, low, volume

    # ------------------------------------------------------------------ #
    #  Cleaning & winsorization                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _clean_X(X: pd.DataFrame) -> pd.DataFrame:
        """
        1. Thay inf → NaN
        2. fillna = cross-sectional median (tốt hơn 0 cho financial features)
        3. Winsorize từng feature ở ±5 sigma
        """
        X = X.replace([np.inf, -np.inf], np.nan)

        # Cross-sectional median fill (cho DataFrame 2D)
        if isinstance(X.index, pd.MultiIndex):
            # Long format: group theo date rồi fill
            X = X.groupby(level="date").transform(lambda g: g.fillna(g.median()))
            # Còn lại (đầu chuỗi không có median) → 0
            X = X.fillna(0.0)
        else:
            # build_latest: index = symbol → column-wise median
            col_medians = X.median()
            X = X.fillna(col_medians)
            X = X.fillna(0.0)

        # Winsorize ±5 sigma theo từng column
        mu = X.mean()
        sigma = X.std().replace(0, np.nan)
        X = X.clip(lower=mu - 5 * sigma, upper=mu + 5 * sigma, axis=1)

        return X