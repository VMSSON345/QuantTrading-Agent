"""API routes cho backtest thủ công."""
from fastapi import APIRouter
from ..schemas.request_schemas import BacktestRequest
from ..schemas.response_schemas import BacktestResponse, MetricsOut
from ...data.data_loader import DataLoader
from ...data.preprocessor import Preprocessor
from ...backtest.backtest_engine import BacktestEngine
from ...backtest.evaluator import Evaluator
from ...utils.code_runner import CodeRunner
from loguru import logger

router = APIRouter()


@router.post("/run", response_model=BacktestResponse)
def run_backtest(req: BacktestRequest):
    try:
        loader = DataLoader()
        data   = Preprocessor(req.start_date, req.end_date).process(loader.get_data())
        engine = BacktestEngine(data)
        runner = CodeRunner()
        alpha_cls = runner.load_alpha_class(req.code)
        signals   = engine.compute_signals(alpha_cls)
        metrics = engine.evaluate(signals, data["close"])
        #metrics   = Evaluator().evaluate(signals, data["close"])
        return BacktestResponse(success=True,
                                metrics=MetricsOut(**metrics.__dict__))
    except Exception as e:
        logger.error(f"backtest error: {e}")
        return BacktestResponse(success=False, error=str(e))