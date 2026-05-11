"""API routes cho Alpha Mining (Inner + Outer Loop)."""
from fastapi import APIRouter
from loguru import logger
from ..schemas.request_schemas import AlphaMineRequest
from ..schemas.response_schemas import AlphaMineResponse, MetricsOut
from ...utils.paths import RAW_DIR
from ...data.data_loader import DataLoader
from ...data.preprocessor import Preprocessor
from ...backtest.backtest_engine import BacktestEngine
from ...backtest.outer_loop import OuterLoop
from ...agents.writer_agent import WriterAgent
from ...agents.judge_agent import JudgeAgent
from ...agents.reviewer_agent import ReviewerAgent
from ...agents.inner_loop import InnerLoop
from ...agents.interpreter_agent import AlphaInterpreterAgent
from ...kb.knowledge_base import KnowledgeBase
from ...utils.code_runner import CodeRunner
from ...kb.alpha_101 import Alpha101KB
from dotenv import load_dotenv
import os

load_dotenv()


router = APIRouter()
_kb = KnowledgeBase()
alpha = Alpha101KB()


def _build_outer_loop(max_inner: int, max_outer: int) -> OuterLoop:
    loader = DataLoader()
    data   = Preprocessor().process(loader.get_data())
    engine = BacktestEngine(data)
    runner = CodeRunner()
    writer = WriterAgent(kb=_kb, alpha101_kb=alpha, api_key = os.getenv("GROQ_WRITE"))
    judge  = JudgeAgent(kb=_kb, api_key = os.getenv("GROQ_JUDGE"))

    inner = InnerLoop(
        writer=writer,
        judge=judge,
        runner=runner,
        max_inner_iter=max_inner,
    )

    reviewer = ReviewerAgent(api_key= os.getenv("GROQ_API_KEY"))
    return OuterLoop(
        inner=inner,
        reviewer=reviewer,
        engine=engine,
        kb=_kb,
        runner=runner,
        max_iter=max_outer,
        min_ic_to_save=0.01,
    )


@router.post("/mine", response_model=AlphaMineResponse)
async def mine_alpha(req: AlphaMineRequest):
    """Chạy Inner + Outer Loop để khai thác alpha từ ý tưởng."""
    try:
        loop   = _build_outer_loop(req.max_inner_iter, req.max_outer_iter)
        result = loop.run(req.idea)
        m      = result.best_metrics or {}
        latex  = result.best_latex or ""

        # Giải thích alpha — truyền latex để interpreter phân tích công thức
        interpretation = ""
        interpretation_payload = {}
        if m and m.get("ic", 0) != 0:
            try:
                interp = AlphaInterpreterAgent()
                symbol = getattr(req, "symbol", "") or ""
                interpretation_payload = interp.interpret(
                    idea=req.idea,
                    symbol=symbol,
                    metrics=m,
                    code=result.best_code or "",
                    latex=latex,
                )
                interpretation = interpretation_payload.get("report_text", "")
            except Exception as e:
                logger.warning(f"Interpreter lỗi: {e}", exc_info=True)
                interpretation = "Không thể tạo phân tích tự động."
                interpretation_payload = {"report_text": interpretation}

        metrics_out = None
        if m:
            metrics_out = MetricsOut(
                ic=m.get("ic", 0),
                sharpe=m.get("sharpe", 0),
                max_drawdown=m.get("max_drawdown", 0),
                win_rate=m.get("win_rate", 0),
                valid_ratio=m.get("valid_ratio", 0),
                n_days=m.get("n_days", 0),
                top_bucket_return=m.get("top_bucket_return", 0),
                bottom_bucket_return=m.get("bottom_bucket_return", 0),
                long_short_spread=m.get("long_short_spread", 0),
            )

        return AlphaMineResponse(
            success=True,
            code=result.best_code or "",
            latex=latex,
            added_to_kb=result.added_to_kb,
            metrics=metrics_out,
            outer_history=result.history,
            interpretation=interpretation,
            interpretation_payload=interpretation_payload,
        )

    except Exception as e:
        logger.error(f"mine_alpha error: {e}", exc_info=True)
        return AlphaMineResponse(success=False, error=str(e))


@router.get("/symbols")
def list_symbols():
    """Lấy danh sách symbols hiện có trong data/raw."""
    files = sorted(RAW_DIR.glob("stock_data_*.csv"))
    syms  = [f.stem.replace("stock_data_", "") for f in files]
    return {"symbols": syms, "total": len(syms)}