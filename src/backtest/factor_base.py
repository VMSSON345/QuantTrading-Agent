import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class AlphaBase(ABC):
    """Base class cho tat ca alpha factors."""
    name          = "AlphaBase"
    window_length = 20
    inputs        = ["close"]

    @abstractmethod
    def calc(self, data: dict) -> pd.DataFrame:
        """
        Tinh alpha signal.
        Args:
            data: dict{"close","open","high","low","volume"} - moi key la DataFrame(date x symbol)
        Returns:
            pd.DataFrame(date x symbol) - gia tri so thuc lien tuc, da fillna(0)
        """
        ...

    def _ret(self, df: pd.DataFrame, n: int = 1) -> pd.DataFrame:
        """Helper: pct_change(n)"""
        return df.pct_change(n)

    def _zscore(self, df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
        """Helper: z-score rolling"""
        mu  = df.rolling(n).mean()
        std = df.rolling(n).std().replace(0, np.nan)
        return (df - mu) / std

    def _rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper: cross-sectional rank [0,1]"""
        return df.rank(axis=1, pct=True)

    def _clip(self, df: pd.DataFrame, lo: float = -3, hi: float = 3) -> pd.DataFrame:
        return df.clip(lo, hi)


# ─── ALPHA MAU SAN (dung truc tiep hoac lam tham khao) ───────

class MomentumAlpha(AlphaBase):
    """Momentum 5 ngay, rank cross-sectional"""
    name = "Momentum5D"
    inputs = ["close"]

    def calc(self, data):
        close = data["close"].ffill().fillna(0)
        signal = self._rank(self._ret(close, 5))
        return signal.fillna(0)


class MeanReversionAlpha(AlphaBase):
    """Mean reversion: z-score am cua gia so voi trung binh 20 ngay"""
    name = "MeanRev20D"
    inputs = ["close"]

    def calc(self, data):
        close = data["close"].ffill().fillna(0)
        signal = -self._zscore(close, 20)
        return signal.fillna(0)


class VolumeZscoreAlpha(AlphaBase):
    """Volume z-score: volume bat thuong duoi bao hieu dong tien lon"""
    name = "VolumeZscore"
    inputs = ["close", "volume"]

    def calc(self, data):
        volume = data["volume"].ffill().fillna(1)
        signal = self._zscore(volume, 20)
        return signal.fillna(0)


class VWAPDevAlpha(AlphaBase):
    """Do lech gia so voi VWAP 10 ngay"""
    name = "VWAPDev10D"
    inputs = ["close", "volume"]

    def calc(self, data):
        close  = data["close"].ffill().fillna(0)
        volume = data["volume"].ffill().fillna(1)
        vwap   = (close * volume).rolling(10).sum() / volume.rolling(10).sum().replace(0, np.nan)
        signal = -(close - vwap) / vwap.replace(0, np.nan)
        return signal.fillna(0)


class MomVolComboAlpha(AlphaBase):
    """Ket hop: Momentum rank * Volume z-score (clip [-2,2])"""
    name = "MomVolCombo"
    inputs = ["close", "volume"]

    def calc(self, data):
        close  = data["close"].ffill().fillna(0)
        volume = data["volume"].ffill().fillna(1)
        mom    = self._rank(self._ret(close, 5))
        vol_z  = self._clip(self._zscore(volume, 10), -2, 2)
        signal = mom * vol_z
        return signal.fillna(0)


class ShortTermRevAlpha(AlphaBase):
    """Dao chieu ngan han: trung binh am cua return 3 ngay gan nhat"""
    name = "ShortTermRev"
    inputs = ["close"]

    def calc(self, data):
        close  = data["close"].ffill().fillna(0)
        signal = -self._ret(close, 1).rolling(3).mean()
        return signal.fillna(0)


# Map ten -> class (backtest engine co the load theo ten)
ALPHA_REGISTRY = {
    "Momentum5D":   MomentumAlpha,
    "MeanRev20D":   MeanReversionAlpha,
    "VolumeZscore": VolumeZscoreAlpha,
    "VWAPDev10D":   VWAPDevAlpha,
    "MomVolCombo":  MomVolComboAlpha,
    "ShortTermRev": ShortTermRevAlpha,
}
