"""API routes cho dự báo giá cổ phiếu."""
from fastapi import APIRouter
from ..schemas.request_schemas import PredictRequest
from ..schemas.response_schemas import PredictResponse
from ...ml.ensemble_predictor import EnsemblePredictor
from ...kb.knowledge_base import KnowledgeBase
from ...data.data_loader import DataLoader
from ...backtest.backtest_engine import BacktestEngine
from ...data.preprocessor import Preprocessor
from loguru import logger

router = APIRouter()
_kb    = KnowledgeBase()


@router.post("/run", response_model=PredictResponse)
def run_prediction(req: PredictRequest):
    try:
        # ===== LOAD DATA =====
        loader = DataLoader()
        data   = Preprocessor().process(loader.get_data())

        engine = BacktestEngine(data)

        # ===== INIT MODEL =====
        pred = EnsemblePredictor(kb=_kb, engine=engine)

        # 🔥 LUÔN train nếu chưa có model
        if req.retrain or pred.model is None:
            pred.train()

        # 🔥 LẤY FULL PAYLOAD (QUAN TRỌNG)
        data_2025   = Preprocessor(start="2025-01-01", end="2025-01-31").process(loader.get_data())
        result = pred.get_ui_payload(req.symbol, data_2025 = data_2025)

        return PredictResponse(**result)

    except Exception as e:
        logger.error(f"predict error: {e}")
        return PredictResponse(
            symbol=req.symbol,
            predicted_return_5d=0.0,
            error=str(e)
        )