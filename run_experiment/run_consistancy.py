import json
from datetime import datetime
from eval import run_eval_once
from src.kb.knowledge_base import KnowledgeBase

BEST_MIN_IC    = 0.01
BEST_MAX_INNER = 3
MAX_OUTER_ITER = 4

TEST_START, TEST_END = "2024-09-01", "2025-04-01"

TEST_IDEAS = "Tôi quan sát thấy biên độ dao động trong ngày (high−low) đang co hẹp dần qua nhiều phiên liên tiếp. Thị trường có thể đang tích lũy trước một biến động lớn. Hãy triển khai một tín hiệu giao dịch dựa trên quan sát này."
LOG_PATH = "experiment_results/consist.jsonl"

kb = KnowledgeBase()



for i in range(10):
    r = run_eval_once(TEST_IDEAS, TEST_START, TEST_END, BEST_MIN_IC, BEST_MAX_INNER, MAX_OUTER_ITER, kb)

    history = r.history  # list các vòng outer: [{k, judge_score, ic, sharpe, ...}, ...]
    first_iter = history[0] if history else None
    last_iter  = history[-1] if history else None
    best_iter  = max(history, key=lambda h: h["ic"]) if history else None

    entry = {
        "timestamp": datetime.now().isoformat(),
        "idea": TEST_IDEAS,
        "config": {"min_ic": BEST_MIN_IC, "max_inner": BEST_MAX_INNER},
        "final_metrics": r.best_metrics,          # metrics của alpha tốt nhất (đã lưu KB nếu qua threshold)             # MỚI
        "best_outer_iter": best_iter,                # MỚI — thường hữu ích hơn "last" để so sánh cải thiện
        "full_history": history,                     # MỚI — toàn bộ để vẽ đường cải thiện nếu cần
    }

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")