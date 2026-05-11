from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class MetricsOut(BaseModel):
    ic: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    valid_ratio: float = 0.0
    n_days: int = 0
    top_bucket_return: float = 0.0
    bottom_bucket_return: float = 0.0
    long_short_spread: float = 0.0


class AlphaOut(BaseModel):
    alpha_id: str
    name: str
    idea: str
    code: str
    metrics: MetricsOut
    judge_score: float
    created_at: str
    review_comment: str = ""


class AlphaMineResponse(BaseModel):
    success: bool
    alpha_id: Optional[str] = None
    code: str = ""
    latex: str = ""
    metrics: Optional[MetricsOut] = None
    added_to_kb: bool = False
    interpretation: str = ""
    interpretation_payload: Dict[str, Any] = {}
    inner_history: List[dict] = []
    outer_history: List[dict] = []
    error: Optional[str] = None


class PredictResponse(BaseModel):
    symbol: str = ""
    predicted_return_5d: float = 0.0
    rank_pct: float = 0.0
    bucket: int = 0
    top_features: List[dict] = []

    train_summary: Dict[str, Any] = {}
    market_snapshot: Dict[str, Any] = {}
    top_picks: List[dict] = []
    bottom_picks: List[dict] = []
    bucket_stats: Dict[str, Any] = {}
    corr_summary: Dict[str, Any] = {}

    top10_compare: List[dict] = []
    accuracy: float = 0.0

    error: Optional[str] = None


class BacktestResponse(BaseModel):
    success: bool
    metrics: Optional[MetricsOut] = None
    error: Optional[str] = None


class KBListResponse(BaseModel):
    total: int
    alphas: List[AlphaOut]
