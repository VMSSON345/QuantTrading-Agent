from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class AlphaMetrics:
    ic: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    valid_ratio: float = 0.0
    n_days: int = 0
    top_bucket_return: float = 0.0
    bottom_bucket_return: float = 0.0
    long_short_spread: float = 0.0


@dataclass
class AlphaRecord:
    alpha_id: str
    name: str
    idea: str
    code: str
    metrics: AlphaMetrics
    latex: str = ""
    judge_score: float = 0.0
    review_comment: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
