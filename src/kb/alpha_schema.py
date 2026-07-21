from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional 
import pandas as pd


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
    icir: float = 0.0          # Information Ratio của IC (Đo lường độ ổn định)
    tic: float = 0.0           # t-statistic của IC (Kiểm định ý nghĩa thống kê, >= 2.0 là tốt)
    ci_lower: float = 0.0      # Cận dưới của khoảng tin cậy 95%
    ci_upper: float = 0.0      # Cận trên của khoảng tin cậy 95%
    
    # --- Dữ liệu chuỗi thời gian ---
    # Sử dụng Optional[pd.Series] vì nếu Alpha fail, nó sẽ trả về None
    ic_ts: dict = field(default_factory=dict)
    


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
