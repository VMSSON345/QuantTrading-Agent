"""API routes cho Knowledge Base Explorer."""
from fastapi import APIRouter, HTTPException
from ..schemas.response_schemas import KBListResponse, AlphaOut, MetricsOut
from ...kb.knowledge_base import KnowledgeBase

router = APIRouter()
_kb    = KnowledgeBase()


@router.get("/list", response_model=KBListResponse)
def list_alphas(min_ic: float = 0.0, top_k: int = 50):
    alphas = _kb.list_all(min_ic=min_ic)[:top_k]
    return KBListResponse(
        total=_kb.count(),
        alphas=[AlphaOut(
            alpha_id=a.alpha_id, name=a.name, idea=a.idea,
            code=a.code,
            metrics=MetricsOut(ic=a.metrics.ic, sharpe=a.metrics.sharpe,
                               max_drawdown=a.metrics.max_drawdown,
                               win_rate=a.metrics.win_rate,
                               valid_ratio=a.metrics.valid_ratio,
                               n_days=a.metrics.n_days,
                               top_bucket_return=a.metrics.top_bucket_return,
                               bottom_bucket_return=a.metrics.bottom_bucket_return,
                               long_short_spread=a.metrics.long_short_spread),
            judge_score=a.judge_score, created_at=a.created_at,
            review_comment=a.review_comment,
        ) for a in alphas]
    )


@router.get("/stats")
def kb_stats():
    total  = _kb.count()
    alphas = _kb.list_all()
    if not alphas:
        return {"total": 0, "avg_ic": 0, "avg_sharpe": 0, "best_ic": 0}
    return {
        "total": total,
        "avg_ic":     round(sum(a.metrics.ic for a in alphas)     / total, 4),
        "avg_sharpe": round(sum(a.metrics.sharpe for a in alphas) / total, 4),
        "best_ic":    round(max(a.metrics.ic for a in alphas), 4),
    }