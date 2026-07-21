"""Hàm dùng chung để chạy alpha mining ở chế độ eval (validation/test) —
   luôn allow_write=False, luôn yêu cầu tường minh start_date/end_date."""
import os
from dotenv import load_dotenv
from src.data.data_loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.backtest.backtest_engine import BacktestEngine
from src.backtest.outer_loop import OuterLoop
from src.agents.writer_agent import WriterAgent
from src.agents.judge_agent import JudgeAgent
from src.agents.reviewer_agent import ReviewerAgent
from src.agents.inner_loop import InnerLoop
from src.kb.knowledge_base import KnowledgeBase
from src.kb.alpha_101 import Alpha101KB
from src.kb.null_kb import NullKB, NullAlpha101KB          # MỚI — cần file này (đã đưa ở câu trước)
from src.utils.code_runner import CodeRunner

load_dotenv()

_alpha101 = Alpha101KB()
_null_alpha101 = NullAlpha101KB()


def run_eval_once(idea: str, start_date: str, end_date: str,
                   min_ic_to_save: float, max_inner_iter: int,
                   max_outer_iter: int, kb: KnowledgeBase,
                   use_main_kb: bool = True,        # MỚI
                   use_alpha101_kb: bool = True):    # MỚI
    """Chạy 1 idea trong chế độ eval — không bao giờ ghi KB."""
    loader = DataLoader()
    data   = Preprocessor(start_date, end_date).process(loader.get_data())
    engine = BacktestEngine(data)
    runner = CodeRunner()

    # Chọn KB thật hoặc KB rỗng tùy cờ ablation — ĐÂY LÀ DÒNG BẠN ĐANG THIẾU
    effective_kb       = kb if use_main_kb else NullKB()
    effective_alpha101 = _alpha101 if use_alpha101_kb else _null_alpha101

    writer = WriterAgent(kb=effective_kb, alpha101_kb=effective_alpha101, api_key=os.getenv("GROQ_WRITE"))
    judge  = JudgeAgent(kb=effective_kb, api_key=os.getenv("GROQ_JUDGE"))
    inner  = InnerLoop(writer=writer, judge=judge, runner=runner, max_inner_iter=max_inner_iter)
    reviewer = ReviewerAgent(api_key=os.getenv("GROQ_API_KEY"))

    loop = OuterLoop(
        inner=inner, reviewer=reviewer, engine=engine, kb=kb, runner=runner,
        max_iter=max_outer_iter,
        min_ic_to_save=min_ic_to_save,
        allow_write=False,
    )
    return loop.run(idea)