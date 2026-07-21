"""Chạy validation: dò (min_ic_to_save, max_inner_iter) trên tập validation.
   KB dùng là KB đã tích lũy từ train (đọc, không ghi)."""
import itertools
import json
from datetime import datetime
from eval import run_eval_once
from src.kb.knowledge_base import KnowledgeBase

# Idea PHẢI khác idea đã dùng ở train — nếu không sẽ chỉ đo khả năng "tái tạo" chứ không đo generalize
VALIDATION_IDEAS = [
    "Tôi quan sát thấy biên độ dao động trong ngày (high−low) đang co hẹp dần qua nhiều phiên liên tiếp. Thị trường có thể đang tích lũy trước một biến động lớn. Hãy triển khai một tín hiệu giao dịch dựa trên quan sát này.",
    "Tôi quan sát thấy giá đóng cửa hôm nay nằm ngoài khoảng dao động (high-low) của ngày hôm trước. Đây có thể là dấu hiệu breakout mạnh theo ngày. Hãy triển khai một tín hiệu giao dịch dựa trên quan sát này.",
    "Tôi quan sát thấy khoảng cách giữa giá mở cửa và giá đóng cửa của phiên trước đang thu hẹp dần theo thời gian. Sự lưỡng lự của thị trường có thể đang gia tăng. Hãy triển khai một tín hiệu giao dịch dựa trên quan sát này.",
    "Tôi quan sát thấy độ biến động (volatility) của cổ phiếu trong 10 ngày gần nhất thấp hơn hẳn so với 60 ngày trước đó. Giai đoạn biến động thấp thường đi trước một đợt biến động mạnh. Hãy triển khai một tín hiệu giao dịch dựa trên quan sát này.",
    "Tôi quan sát thấy tỷ lệ khối lượng giao dịch buổi đầu phiên so với cả phiên đang tăng dần qua các ngày. Dòng tiền có thể đang vào sớm hơn trong ngày. Hãy triển khai một tín hiệu giao dịch dựa trên quan sát này.",
    "Tôi quan sát thấy chuỗi các mức đáy (low) đang tăng dần trong khi giá đóng cửa chưa vượt đỉnh cũ. Đây có thể là dấu hiệu tích lũy trước khi bứt phá. Hãy triển khai một tín hiệu giao dịch dựa trên quan sát này.",
    "Tôi quan sát thấy độ lệch chuẩn của khối lượng giao dịch tăng mạnh trong khi giá gần như không đổi. Có sự bất thường giữa dòng tiền và biến động giá. Hãy triển khai một tín hiệu giao dịch dựa trên quan sát này.",
    "Tôi quan sát thấy số ngày tăng giá liên tiếp trong 10 phiên gần nhất nhiều hơn hẳn số ngày giảm. Đà tăng có thể đang chiếm ưu thế rõ rệt. Hãy triển khai một tín hiệu giao dịch dựa trên quan sát này.",
    "Tôi quan sát thấy giá đóng cửa liên tục nằm giữa giá mở và giá cao nhất trong ngày (không chạm gần giá thấp). Lực mua có thể đang kiểm soát toàn phiên. Hãy triển khai một tín hiệu giao dịch dựa trên quan sát này.",
    "Tôi quan sát thấy tương quan giữa return của cổ phiếu và return trung bình toàn thị trường đang suy yếu dần. Cổ phiếu có thể đang tách khỏi xu hướng chung. Hãy triển khai một tín hiệu giao dịch dựa trên quan sát này.",
]

MIN_IC_GRID    = [0.005, 0.01, 0.02]
MAX_INNER_GRID = [3, 5, 8]

VAL_START, VAL_END = "2024-01-01", "2024-08-31"   # cùng khung train, vì KB đã đóng băng nên không leak
MAX_OUTER_ITER = 5   # giữ nguyên bằng với train để so sánh công bằng

kb = KnowledgeBase()   # đọc KB đã tích lũy từ train, KHÔNG ghi gì trong suốt validation

results = []
for min_ic, max_inner in itertools.product(MIN_IC_GRID, MAX_INNER_GRID):
    sharpe_scores, ic_scores = [], []
    for idea in VALIDATION_IDEAS:
        r = run_eval_once(idea, VAL_START, VAL_END, min_ic, max_inner, MAX_OUTER_ITER, kb)
        if r.best_metrics:
            sharpe_scores.append(r.best_metrics.get("sharpe", 0))
            ic_scores.append(r.best_metrics.get("ic", 0))

    avg_sharpe = sum(sharpe_scores) / len(sharpe_scores) if sharpe_scores else None
    avg_ic     = sum(ic_scores) / len(ic_scores) if ic_scores else None
    results.append({"min_ic": min_ic, "max_inner": max_inner,
                     "avg_sharpe": avg_sharpe, "avg_ic": avg_ic})
    print(f"min_ic={min_ic} max_inner={max_inner} -> avg_sharpe={avg_sharpe} avg_ic={avg_ic}")

with open("experiment_results/validation_grid.json", "w", encoding="utf-8") as f:
    json.dump({"timestamp": datetime.now().isoformat(), "results": results}, f, indent=2, ensure_ascii=False)

best = max([r for r in results if r["avg_sharpe"] is not None], key=lambda x: x["avg_sharpe"])
print("\n>>> Config tốt nhất:", best)