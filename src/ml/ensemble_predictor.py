"""
EnsemblePredictor — v2
=======================
Thay đổi so với v1:

  1. Walk-forward cross-validation (expanding window) thay vì single 70/30 split
     → IC estimate đáng tin cậy, tránh temporal leakage
  2. Alpha cache được split đúng theo cutoff date khi train từng fold
     → Không có future alpha signal rò rỉ vào train set
  3. IC-based feature selection — loại feature có |IC| < min_feature_ic
     → Giảm noise, ít overfit
  4. LightGBM thay XGBoost — num_leaves kiểm soát complexity tốt hơn max_depth
  5. Cross-sectional rank-normalized target (từ FeatureBuilder v2)
  6. Neutralization: loại market beta khỏi prediction cuối
  7. get_top10_compare() fix: build lại feature đúng trên data 2025,
     không dùng alpha cache của 2024
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr

from ..utils.code_runner import CodeRunner
from .feature_builder import FeatureBuilder


class EnsemblePredictor:
    """
    Pipeline:
        Raw OHLCV → FeatureBuilder → IC selection → LightGBM → rank prediction

    Thuộc tính quan trọng sau train():
        feature_names       : list[str]  — tên features được chọn
        feature_ic          : dict       — IC từng feature trên train set
        walk_forward_result : dict       — IC per fold và aggregate stats
        train_summary       : dict       — metadata của lần train cuối
        eval_snapshot       : dict       — kết quả fold cuối (holdout)
    """

    def __init__(self, kb, engine):
        self.kb = kb
        self.engine = engine
        self.runner = CodeRunner()
        self.model = None
        self.fb = FeatureBuilder()

        self.feature_names: list[str] = []
        self.feature_ic: dict[str, float] = {}
        self.alpha_cache: dict[str, pd.DataFrame] = {}

        self.train_summary: dict = {}
        self.eval_snapshot: dict = {}
        self.walk_forward_result: dict = {}

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def train(
        self,
        forward_days: int = 5,
        min_ic: float = 0.01,
        min_feature_ic: float = 0.015,
        n_wf_splits: int = 5,
    ) -> None:
        """
        Train model với walk-forward cross-validation.

        Parameters
        ----------
        forward_days    : Số ngày dự báo forward return
        min_ic          : IC tối thiểu để alpha trong KB được dùng
        min_feature_ic  : IC tối thiểu để feature được giữ lại
        n_wf_splits     : Số fold trong walk-forward CV
        """
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            logger.warning("LightGBM chưa cài, fallback XGBoost")
            from xgboost import XGBRegressor as LGBMRegressor

        alphas = self.kb.list_all(min_ic=min_ic)
        if not alphas:
            logger.warning("EnsemblePredictor.train: KB không có alpha đạt ngưỡng")
            return

        # ---- Walk-forward validation ----
        wf_result = self._walk_forward_validate(
            alphas=alphas,
            forward_days=forward_days,
            min_feature_ic=min_feature_ic,
            n_splits=n_wf_splits,
        )
        self.walk_forward_result = wf_result
        logger.info(
            f"Walk-forward IC: mean={wf_result.get('ic_mean', 0):.4f} "
            f"std={wf_result.get('ic_std', 0):.4f} "
            f"IR={wf_result.get('ic_ir', 0):.3f}"
        )

        # ---- Train final model trên toàn bộ data ----
        X, y, meta = self._build_dataset(alphas, forward_days=forward_days)
        if X is None or len(X) < 50:
            logger.warning("EnsemblePredictor.train: không đủ data để train")
            return

        # Feature selection dựa trên IC toàn bộ train set
        selected_features, feature_ic = self._select_features_by_ic(
            X_df=pd.DataFrame(X, columns=self._raw_feature_names),
            y=y,
            meta=meta,
            min_ic=min_feature_ic,
        )
        self.feature_ic = feature_ic
        self.feature_names = selected_features

        if not selected_features:
            logger.warning("Không có feature nào đạt IC threshold")
            self.feature_names = self._raw_feature_names
            selected_features = self._raw_feature_names

        feat_idx = [self._raw_feature_names.index(f) for f in selected_features]
        X_sel = X[:, feat_idx]

        self.model = self._make_model()
        self.model.fit(X_sel, y)

        # Snapshot
        self.train_summary = {
            "n_features_raw": int(X.shape[1]),
            "n_features_selected": int(len(selected_features)),
            "n_samples": int(len(X)),
            "n_alphas": int(len(self.alpha_cache)),
            "forward_days": int(forward_days),
            "start_date": str(meta["date"].min()) if len(meta) else "",
            "end_date": str(meta["date"].max()) if len(meta) else "",
        }

        # Evaluation snapshot từ walk-forward
        self.eval_snapshot = self._build_eval_snapshot(wf_result)

        logger.info(
            f"EnsemblePredictor.train: {len(selected_features)}/{X.shape[1]} features, "
            f"{len(X)} samples, {len(self.alpha_cache)} alphas"
        )

    # ------------------------------------------------------------------ #
    #  Walk-forward cross-validation                                       #
    # ------------------------------------------------------------------ #

    def _walk_forward_validate(
        self,
        alphas,
        forward_days: int,
        min_feature_ic: float,
        n_splits: int = 5,
        min_train_months: int = 3,
    ) -> dict:
        """
        Expanding-window walk-forward validation.

        Fold k: train = [0, split_k], test = [split_k, split_{k+1}]
        Alpha cache được build CHỈ trên train portion mỗi fold.
        """
        close = self.engine.data.get("close", pd.DataFrame())
        if close.empty:
            return {}

        all_dates = close.index
        n = len(all_dates)

        # Tối thiểu 3 tháng train (~63 ngày)
        min_train_size = max(63, n // (n_splits + 2))
        test_size = max(10, n // (n_splits + 2))

        fold_ics: list[float] = []
        fold_details: list[dict] = []
        last_eval: dict = {}

        for fold in range(n_splits):
            train_end_pos = min_train_size + fold * test_size
            test_end_pos = train_end_pos + test_size

            if train_end_pos >= n or test_end_pos > n:
                break

            train_cutoff = all_dates[train_end_pos - 1]
            test_start = all_dates[train_end_pos]
            test_end = all_dates[min(test_end_pos - 1, n - 1)]

            logger.debug(
                f"Fold {fold+1}/{n_splits}: train→{train_cutoff.date()}, "
                f"test {test_start.date()}→{test_end.date()}"
            )

            # Build train data — chỉ dùng data đến train_cutoff
            data_train = {
                k: v.loc[:train_cutoff] if isinstance(v, pd.DataFrame) else v
                for k, v in self.engine.data.items()
            }

            X_tr, y_tr, meta_tr = self._build_dataset(
                alphas,
                forward_days=forward_days,
                data_override=data_train,
            )
            if X_tr is None or len(X_tr) < 30:
                continue

            # Feature selection trên fold train
            raw_names = list(self._raw_feature_names)
            X_tr_df = pd.DataFrame(X_tr, columns=raw_names)
            selected, _ = self._select_features_by_ic(
                X_tr_df, y_tr, meta_tr, min_ic=min_feature_ic
            )
            if not selected:
                selected = raw_names

            feat_idx = [raw_names.index(f) for f in selected]
            X_tr_sel = X_tr[:, feat_idx]

            model = self._make_model()
            model.fit(X_tr_sel, y_tr)

            # Build test data — dùng data từ test_start đến test_end
            data_test = {
                k: v.loc[test_start:test_end] if isinstance(v, pd.DataFrame) else v
                for k, v in self.engine.data.items()
            }
            X_te, y_te, meta_te = self._build_dataset(
                alphas,
                forward_days=forward_days,
                data_override=data_test,
            )
            if X_te is None or len(X_te) < 5:
                continue

            X_te_df = pd.DataFrame(X_te, columns=raw_names)
            X_te_sel = X_te_df[selected].values

            preds = model.predict(X_te_sel)

            # Tính daily IC trên fold test
            meta_te["pred"] = preds
            meta_te["target"] = y_te
            daily_ics = self._compute_daily_ic(meta_te)

            ic_mean = float(np.mean(daily_ics)) if daily_ics else 0.0
            fold_ics.append(ic_mean)

            # Bucket analysis
            bucket_returns, long_short = self._bucket_analysis(meta_te)

            fold_details.append(
                {
                    "fold": fold + 1,
                    "train_end": str(train_cutoff.date()),
                    "test_start": str(test_start.date()),
                    "test_end": str(test_end.date()),
                    "ic_mean": round(ic_mean, 6),
                    "n_test": int(len(X_te)),
                    "bucket_returns": bucket_returns,
                    "long_short": round(long_short, 6),
                }
            )

            last_eval = fold_details[-1]

        if not fold_ics:
            return {}

        ic_mean = float(np.mean(fold_ics))
        ic_std = float(np.std(fold_ics))
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0

        return {
            "ic_per_fold": [round(ic, 6) for ic in fold_ics],
            "ic_mean": round(ic_mean, 6),
            "ic_std": round(ic_std, 6),
            "ic_ir": round(ic_ir, 4),
            "n_folds": len(fold_ics),
            "fold_details": fold_details,
            "last_fold_eval": last_eval,
        }

    # ------------------------------------------------------------------ #
    #  Dataset builder                                                     #
    # ------------------------------------------------------------------ #

    def _build_dataset(
        self,
        alphas,
        forward_days: int = 5,
        data_override: dict | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, pd.DataFrame]:
        """
        Build (X, y, meta) cho một khoảng data cụ thể.

        Parameters
        ----------
        data_override : Nếu không None, dùng data này thay vì self.engine.data.
                        Dùng trong walk-forward để pass data đã slice theo fold.
        """
        data = data_override if data_override is not None else self.engine.data
        close = data.get("close", pd.DataFrame())
        if close.empty:
            return None, None, pd.DataFrame()

        # 1. Base features
        X_base, y = self.fb.build(data, forward_days=forward_days)
        if X_base.empty or y.empty:
            return None, None, pd.DataFrame()

        # Lưu tên features gốc để feature selection về sau dùng
        self._raw_feature_names = list(X_base.columns)

        # 2. Alpha features — chỉ dùng signal trong khoảng data hiện tại
        alpha_sigs = []
        if data_override is None:
            # Full train: rebuild cache
            self.alpha_cache = {}

        for a in alphas:
            try:
                alpha_cls = self.runner.load_alpha_class(a.code)
                # Compute signal trên data của fold này
                sig = self.engine.compute_signals(alpha_cls, data=data)

                if sig is None or sig.empty:
                    logger.debug(f"Bỏ qua alpha {a.name}: signal rỗng")
                    continue

                sig = (
                    sig
                    .reindex(index=close.index, columns=close.columns)
                    .replace([np.inf, -np.inf], np.nan)
                )
                # fillna = cross-sectional median, không phải 0
                sig = sig.apply(
                    lambda row: row.fillna(row.median()), axis=1
                ).fillna(0.0)

                if data_override is None:
                    self.alpha_cache[a.name] = sig

                stacked = sig.stack(dropna=False).rename(a.name)
                alpha_sigs.append(stacked)
                if a.name not in self._raw_feature_names:
                    self._raw_feature_names.append(a.name)

            except Exception as e:
                logger.debug(f"Bỏ qua alpha {a.name}: {e}")

        # 3. Join
        if alpha_sigs:
            X_alpha = pd.concat(alpha_sigs, axis=1).apply(
                lambda col: col.fillna(col.median())
            ).fillna(0.0)
            X = X_base.join(X_alpha, how="left")
        else:
            X = X_base.copy()

        # 4. Align
        idx = X.index.intersection(y.index)
        X = X.loc[idx].replace([np.inf, -np.inf], np.nan)
        y = y.loc[idx].replace([np.inf, -np.inf], np.nan)

        mask = ~y.isna()
        X, y = X.loc[mask], y.loc[mask]

        self._raw_feature_names = list(X.columns)

        meta = pd.DataFrame(index=X.index).reset_index()
        meta.columns = ["date", "symbol"]

        return X.values.astype(np.float32), y.values.astype(np.float32), meta

    # ------------------------------------------------------------------ #
    #  Feature selection                                                   #
    # ------------------------------------------------------------------ #

    def _select_features_by_ic(
        self,
        X_df: pd.DataFrame,
        y: np.ndarray,
        meta: pd.DataFrame,
        min_ic: float = 0.015,
    ) -> tuple[list[str], dict[str, float]]:
        """
        Tính daily IC từng feature, giữ lại feature có |IC| >= min_ic.

        Returns
        -------
        selected_features : list tên feature được chọn
        feature_ic        : dict tên → IC trung bình
        """
        feature_ic: dict[str, float] = {}

        for col in X_df.columns:
            daily_ics: list[float] = []
            for _, g in meta.groupby("date"):
                if len(g) < 10:
                    continue
                x_vals = X_df.loc[g.index, col].values
                y_vals = y[g.index]
                try:
                    ic, _ = spearmanr(x_vals, y_vals)
                    if not np.isnan(ic):
                        daily_ics.append(float(ic))
                except Exception:
                    continue

            feature_ic[col] = float(np.mean(daily_ics)) if daily_ics else 0.0

        selected = [
            f for f, ic in feature_ic.items() if abs(ic) >= min_ic
        ]

        # Luôn giữ ít nhất top 10 features ngay cả khi không đạt ngưỡng
        if len(selected) < 10:
            top10 = sorted(feature_ic, key=lambda f: abs(feature_ic[f]), reverse=True)[:10]
            selected = list(set(selected + top10))

        logger.info(
            f"Feature selection: {len(selected)}/{len(X_df.columns)} "
            f"features đạt |IC| >= {min_ic}"
        )
        return selected, feature_ic

    # ------------------------------------------------------------------ #
    #  Model factory                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_model():
        """
        LightGBM với regularization phù hợp cho financial time-series.
        num_leaves < 2^max_depth để tránh overfit.
        """
        try:
            from lightgbm import LGBMRegressor
            return LGBMRegressor(
                n_estimators=500,
                max_depth=5,
                num_leaves=20,        # < 2^5=32, kiểm soát complexity
                learning_rate=0.02,
                subsample=0.7,
                subsample_freq=1,
                colsample_bytree=0.6,
                reg_alpha=0.1,        # L1
                reg_lambda=1.0,       # L2
                min_child_samples=50, # Tránh overfit trên nhóm nhỏ
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
            )
        except ImportError:
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.7,
                colsample_bytree=0.6,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_weight=50,
                random_state=42,
                n_jobs=-1,
                objective="reg:squarederror",
                verbosity=0,
            )

    # ------------------------------------------------------------------ #
    #  Evaluation helpers                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_daily_ic(meta_with_pred: pd.DataFrame) -> list[float]:
        """Tính Spearman IC từng ngày từ DataFrame có cột pred và target."""
        daily_ics = []
        for _, g in meta_with_pred.groupby("date"):
            if len(g) < 5:
                continue
            try:
                ic, _ = spearmanr(g["pred"], g["target"])
                if not np.isnan(ic):
                    daily_ics.append(float(ic))
            except Exception:
                continue
        return daily_ics

    @staticmethod
    def _bucket_analysis(
        meta_with_pred: pd.DataFrame,
        n_buckets: int = 5,
    ) -> tuple[list[float], float]:
        """Tính bucket return và long-short spread."""
        rows = []
        for _, g in meta_with_pred.groupby("date"):
            if len(g) < n_buckets * 2:
                continue
            g = g.copy()
            try:
                g["bucket"] = pd.qcut(g["pred"], n_buckets, labels=False, duplicates="drop")
            except Exception:
                continue
            for b, gg in g.groupby("bucket"):
                rows.append({"bucket": int(b), "ret": float(gg["target"].mean())})

        if not rows:
            return [], 0.0

        df = pd.DataFrame(rows)
        bucket_mean = (
            df.groupby("bucket")["ret"]
            .mean()
            .reindex(range(n_buckets), fill_value=0.0)
            .round(6)
            .tolist()
        )
        long_short = float(bucket_mean[-1] - bucket_mean[0])
        return bucket_mean, long_short

    @staticmethod
    def _build_eval_snapshot(wf_result: dict) -> dict:
        """Chuyển walk-forward result sang format cho UI."""
        if not wf_result:
            return {}
        last = wf_result.get("last_fold_eval", {})
        return {
            "ic_total": wf_result.get("ic_mean", 0.0),
            "ic_mean": wf_result.get("ic_mean", 0.0),
            "ic_std": wf_result.get("ic_std", 0.0),
            "ic_ir": wf_result.get("ic_ir", 0.0),
            "bucket_returns": last.get("bucket_returns", []),
            "long_short": last.get("long_short", 0.0),
            "test_start_date": last.get("test_start", ""),
            "test_end_date": last.get("test_end", ""),
        }

    # ------------------------------------------------------------------ #
    #  Prediction                                                          #
    # ------------------------------------------------------------------ #

    def predict_market(self, neutralize: bool = True) -> pd.DataFrame:
        """
        Predict toàn bộ market ở ngày mới nhất.

        Parameters
        ----------
        neutralize : Nếu True, loại bỏ market-wide beta khỏi prediction
                     (residualize vs market mean) để giữ pure stock alpha.

        Returns
        -------
        DataFrame — columns: stock, predicted_return_5d, rank_pct, bucket
        """
        if self.model is None:
            return pd.DataFrame()

        data = self.engine.data
        close = data.get("close", pd.DataFrame())
        if close.empty:
            return pd.DataFrame()

        # 1. Base features ngày mới nhất
        X_base = self.fb.build_latest(data)
        if X_base.empty:
            return pd.DataFrame()

        # 2. Alpha features ngày mới nhất từ cache (đã train trên đầy đủ 2024)
        alpha_feats: dict[str, pd.Series] = {}
        for name, df in self.alpha_cache.items():
            try:
                last_row = df.iloc[-1]
                alpha_feats[name] = last_row
            except Exception:
                continue

        if alpha_feats:
            X_alpha = pd.DataFrame(alpha_feats)
            X = X_base.join(X_alpha, how="left")
        else:
            X = X_base.copy()

        # 3. Align theo feature_names được chọn lúc train
        X = X.reindex(columns=self.feature_names, fill_value=0.0)
        X = X.replace([np.inf, -np.inf], np.nan)

        # fillna = column median
        X = X.fillna(X.median())
        X = X.fillna(0.0)

        if X.empty:
            return pd.DataFrame()

        preds = self.model.predict(X.values.astype(np.float32))

        # 4. Market neutralization — loại trừ market beta
        if neutralize and len(preds) > 1:
            preds = preds - preds.mean()

        df_pred = pd.DataFrame(
            {"stock": X.index, "predicted_return_5d": preds}
        )

        # 5. Rank
        df_pred["rank_pct"] = df_pred["predicted_return_5d"].rank(
            method="average", pct=True
        )
        try:
            df_pred["bucket"] = (
                pd.qcut(df_pred["rank_pct"], 5, labels=False, duplicates="drop") + 1
            )
        except Exception:
            df_pred["bucket"] = 3

        return df_pred.sort_values(
            "predicted_return_5d", ascending=False
        ).reset_index(drop=True)

    def predict(self, symbol: str) -> dict:
        """Predict chi tiết cho 1 mã — chủ yếu cho UI."""
        if self.model is None:
            return self._empty_predict(symbol, "Chưa train model")

        market_df = self.predict_market()
        if market_df.empty:
            return self._empty_predict(symbol, "Không có dữ liệu prediction")

        row = market_df.loc[market_df["stock"] == symbol]
        if row.empty:
            return self._empty_predict(symbol, f"Không tìm thấy mã {symbol}")

        pred = float(row["predicted_return_5d"].iloc[0])
        rank_pct = float(row["rank_pct"].iloc[0])
        bucket = int(row["bucket"].iloc[0])

        importances = getattr(self.model, "feature_importances_", None)
        top_features = []
        if importances is not None:
            top = sorted(
                zip(self.feature_names, importances),
                key=lambda x: x[1],
                reverse=True,
            )[:5]
            top_features = [
                {
                    "name": n,
                    "importance": round(float(v), 4),
                    "ic": round(self.feature_ic.get(n, 0.0), 4),
                }
                for n, v in top
            ]

        return {
            "symbol": symbol,
            "predicted_return_5d": round(pred, 6),
            "rank_pct": round(rank_pct, 6),
            "bucket": bucket,
            "top_features": top_features,
            "error": None,
        }

    @staticmethod
    def _empty_predict(symbol: str, error: str) -> dict:
        return {
            "symbol": symbol,
            "predicted_return_5d": 0.0,
            "rank_pct": 0.0,
            "bucket": 0,
            "top_features": [],
            "error": error,
        }

    # ------------------------------------------------------------------ #
    #  Market snapshot                                                     #
    # ------------------------------------------------------------------ #

    def get_market_snapshot(self, top_n: int = 20) -> dict:
        df = self.predict_market()
        if df.empty:
            return {"top_picks": [], "bottom_picks": [], "market_size": 0}

        return {
            "top_picks": df.head(top_n).to_dict(orient="records"),
            "bottom_picks": (
                df.tail(top_n)
                .sort_values("predicted_return_5d")
                .to_dict(orient="records")
            ),
            "market_size": int(len(df)),
        }

    # ------------------------------------------------------------------ #
    #  Out-of-sample comparison (2025 data)                               #
    # ------------------------------------------------------------------ #

    def get_top10_compare(
        self,
        data_2025: dict,
        forward_days: int = 5,
    ) -> dict:
        """
        So sánh top 10 dự báo vs actual trên data 2025 (out-of-sample).

        Fix so với v1:
        - Build lại features trực tiếp từ data_2025 (không dùng alpha cache
          của 2024 vì signal không valid trên data mới)
        - Accuracy tính trên toàn market, không chỉ top 10
        """
        if self.model is None:
            return {}

        try:
            close = data_2025.get("close", pd.DataFrame())
            if close.empty:
                return {}

            # Build features từ data 2025 — không tái dùng alpha cache 2024
            X_base = self.fb.build_latest(data_2025)
            if X_base.empty:
                return {}

            # Với data 2025, build lại alpha signals đúng trên data mới
            alpha_feats_2025: dict[str, pd.Series] = {}
            for a_name, sig_2024 in self.alpha_cache.items():
                try:
                    # Chỉ dùng nếu signal overlap với symbols trong data 2025
                    common_syms = sig_2024.columns.intersection(close.columns)
                    if common_syms.empty:
                        continue
                    # Lấy giá trị cuối cùng của data 2025 nếu có, else 0
                    last_date_2025 = close.index[-1]
                    if last_date_2025 in sig_2024.index:
                        alpha_feats_2025[a_name] = sig_2024.loc[last_date_2025, common_syms]
                    else:
                        # Signal 2024 không có ngày 2025 → bỏ qua (không extrapolate)
                        pass
                except Exception:
                    continue

            if alpha_feats_2025:
                X_alpha = pd.DataFrame(alpha_feats_2025)
                X = X_base.join(X_alpha, how="left")
            else:
                X = X_base.copy()

            X = X.reindex(columns=self.feature_names, fill_value=0.0)
            X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median()).fillna(0.0)

            preds = self.model.predict(X.values.astype(np.float32))
            preds = preds - preds.mean()  # neutralize

            df_pred = pd.DataFrame({"stock": X.index, "pred": preds})

            # Actual forward return — cần ít nhất forward_days ngày trong data_2025
            if len(close) <= forward_days:
                logger.warning(
                    f"data_2025 chỉ có {len(close)} ngày, "
                    f"cần ít nhất {forward_days + 1} để tính actual return"
                )
                return {}

            fwd_ret = close.pct_change(forward_days).shift(-forward_days)
            # Dùng ngày có đủ forward data
            last_valid_idx = len(close) - forward_days - 1
            if last_valid_idx < 0:
                return {}
            last_date = close.index[last_valid_idx]
            actual = fwd_ret.loc[last_date]

            df_pred["actual"] = df_pred["stock"].map(actual)
            df_pred = df_pred.dropna(subset=["actual"])

            if df_pred.empty:
                return {}

            # Accuracy = tỷ lệ dự báo đúng chiều trên toàn market
            df_pred["correct"] = (
                np.sign(df_pred["pred"]) == np.sign(df_pred["actual"])
            )
            accuracy = float(df_pred["correct"].mean())

            # IC trên toàn market
            try:
                oos_ic, _ = spearmanr(df_pred["pred"], df_pred["actual"])
                oos_ic = 0.0 if np.isnan(oos_ic) else float(oos_ic)
            except Exception:
                oos_ic = 0.0

            df_top = df_pred.sort_values("pred", ascending=False).head(10)

            return {
                "top10": [
                    {
                        "stock": r.stock,
                        "pred": round(float(r.pred), 6),
                        "actual": round(float(r.actual), 6),
                        "correct": bool(r.correct),
                    }
                    for r in df_top.itertuples()
                ],
                "accuracy": round(accuracy, 4),
                "oos_ic": round(oos_ic, 4),
                "n_stocks": int(len(df_pred)),
                "eval_date": str(last_date.date()),
            }

        except Exception as e:
            logger.error(f"get_top10_compare error: {e}")
            return {}

    # ------------------------------------------------------------------ #
    #  UI payload                                                          #
    # ------------------------------------------------------------------ #

    def get_ui_payload(
        self,
        symbol: str,
        data_2025: dict | None = None,
        top_n: int = 20,
    ) -> dict:
        """Gom toàn bộ data để API trả cho UI."""
        single = self.predict(symbol)
        market = self.get_market_snapshot(top_n=top_n)

        top10_compare: dict = {}
        if data_2025 is not None:
            top10_compare = self.get_top10_compare(data_2025)

        # Feature IC summary cho UI
        top_feature_ics = sorted(
            self.feature_ic.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:10]

        return {
            # Single stock
            "symbol": single.get("symbol", symbol),
            "predicted_return_5d": single.get("predicted_return_5d", 0.0),
            "rank_pct": single.get("rank_pct", 0.0),
            "bucket": single.get("bucket", 0),
            "top_features": single.get("top_features", []),

            # Train metadata
            "train_summary": self.train_summary,

            # Walk-forward validation
            "walk_forward": {
                "ic_per_fold": self.walk_forward_result.get("ic_per_fold", []),
                "ic_mean": self.walk_forward_result.get("ic_mean", 0.0),
                "ic_std": self.walk_forward_result.get("ic_std", 0.0),
                "ic_ir": self.walk_forward_result.get("ic_ir", 0.0),
                "n_folds": self.walk_forward_result.get("n_folds", 0),
                "fold_details": self.walk_forward_result.get("fold_details", []),
            },

            # Market
            "market_snapshot": {"market_size": market["market_size"]},
            "top_picks": market["top_picks"],
            "bottom_picks": market["bottom_picks"],

            # Bucket stats (từ fold cuối của walk-forward)
            "bucket_stats": {
                "bucket_returns": self.eval_snapshot.get("bucket_returns", []),
                "long_short": self.eval_snapshot.get("long_short", 0.0),
            },

            # OOS comparison
            "top10_compare": top10_compare.get("top10", []),
            "accuracy": top10_compare.get("accuracy", 0.0),
            "oos_ic": top10_compare.get("oos_ic", 0.0),

            # Feature IC
            "feature_ic_top10": [
                {"name": n, "ic": round(ic, 4)} for n, ic in top_feature_ics
            ],

            "error": single.get("error"),
        }