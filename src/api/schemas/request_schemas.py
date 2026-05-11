from pydantic import BaseModel, Field

class AlphaMineRequest(BaseModel):
    idea: str = Field(..., min_length=10, max_length=500)
    max_inner_iter: int = Field(5, ge=1, le=5)
    max_outer_iter: int = Field(5, ge=1, le=5)

class PredictRequest(BaseModel):
    symbol: str
    retrain: bool = False

class BacktestRequest(BaseModel):
    code: str
    start_date: str = "2024-01-01"
    end_date: str   = "2024-12-31"

class KBFilterRequest(BaseModel):
    min_ic: float     = 0.0
    min_sharpe: float = 0.0
    top_k: int        = Field(20, ge=1, le=100)
