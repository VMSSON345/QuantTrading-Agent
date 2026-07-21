from dataclasses import dataclass, field
from loguru import logger
from .backtest_engine import BacktestEngine
from ..agents.inner_loop import InnerLoop
from ..agents.reviewer_agent import ReviewerAgent
from ..kb.knowledge_base import KnowledgeBase
from ..kb.alpha_schema import AlphaRecord, AlphaMetrics
from ..utils.code_runner import CodeRunner
import uuid


@dataclass
class OuterResult:
    best_code:    str  = ""
    best_latex:   str  = ""
    best_metrics: dict = field(default_factory=dict)
    added_to_kb:  bool = False
    history:      list = field(default_factory=list)


class OuterLoop:
    def __init__(self, inner, reviewer, engine, kb, runner, max_iter=5, min_ic_to_save=0.01, allow_write=True):
        self.inner        = inner
        self.reviewer     = reviewer
        self.engine       = engine
        self.kb           = kb
        self.runner       = runner
        self.max_iter     = max_iter
        self.min_ic_save  = min_ic_to_save
        self.allow_write   = allow_write

    def run(self, idea: str) -> OuterResult:
        result   = OuterResult()
        feedback = ""
        best_ic  = -999.0
        best_judge_score   = 0.0
        best_review_comment = ""

        for i in range(self.max_iter):
            logger.info(f"Outer iter {i+1}/{self.max_iter}")
            inner_out = self.inner.run(idea, feedback=feedback)
            code  = inner_out.final_code
            latex = inner_out.final_latex

            if not code:
                logger.warning("Inner loop khong sinh duoc code")
                continue

            metrics = {}
            try:
                cls     = self.runner.load_alpha_class(code)
                signals = self.engine.compute_signals(cls)
                if signals.empty:
                    raise ValueError("compute_signals tra ve rong")
                metrics = self.engine.evaluate(signals, forward_days=5)
                logger.info(f"  IC={metrics['ic']:.4f} Sharpe={metrics['sharpe']:.4f}")
            except Exception as e:
                logger.error(f"Outer backtest loi: {e}")
                metrics = BacktestEngine._zero_metrics()

            feedback = self.reviewer.review(code, metrics)
            result.history.append({
                "k":            i + 1,
                "judge_score":  inner_out.final_score,   # MỚI — điểm Judge (0-1) của code sinh ra ở vòng này
                "ic":           metrics.get("ic", 0),
                "sharpe":       metrics.get("sharpe", 0),
                "icir":         metrics.get("icir", 0),   # MỚI — tiện cho phân tích sâu hơn sau này
                "review":       feedback[:120],
            })

            if metrics.get("ic", 0) > best_ic:
                best_ic              = metrics["ic"]
                result.best_code     = code
                result.best_latex    = latex
                result.best_metrics  = metrics
                # FIX: lưu judge_score và review_comment từ inner loop
                best_judge_score     = inner_out.final_score
                best_review_comment  = feedback

        if result.best_code and best_ic >= self.min_ic_save:
            m = result.best_metrics
            print(m.get("icir", 0))
            print(m.get("tic", 0))
            print(m.get("ic_year", 0))
            rec = AlphaRecord(
                alpha_id=str(uuid.uuid4())[:8],
                # FIX: tên hiển thị 40 ký tự, idea lưu đầy đủ
                name=f"Alpha_{idea[:40].replace(' ', '_')}",
                idea=idea,
                code=result.best_code,
                latex=result.best_latex,
                metrics=AlphaMetrics(
                    ic=m.get("ic", 0),
                    sharpe=m.get("sharpe", 0),
                    max_drawdown=m.get("max_drawdown", 0),
                    win_rate=m.get("win_rate", 0),
                    valid_ratio=m.get("valid_ratio", 0),
                    n_days=m.get("n_days", 0),
                    icir=m.get("icir", 0),
                    tic=m.get("tic", 0),
                    ci_lower=m.get("ci_lower", 0),
                    ci_upper=m.get("ci_upper", 0),
                    ic_ts=m.get("ic_year", 0),
                ),
                judge_score=round(best_judge_score * 10, 2),   # FIX: 0.7 → 7.0/10
                review_comment=best_review_comment[:300],
            )
            if self.allow_write:                                       # MỚI — bọc lại
                self.kb.add(rec)
                result.added_to_kb = True
                logger.success(f"Luu KB: IC={best_ic:.4f}")
            else:
                logger.info(f"[EVAL MODE] Khong ghi KB (IC={best_ic:.4f} du nguong)")
        else:
            logger.warning(f"IC={best_ic:.4f} < nguong, khong luu")

        return result