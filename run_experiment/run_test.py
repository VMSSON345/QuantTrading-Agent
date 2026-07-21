"""Chạy test cuối cùng — chỉ chạy 1 lần với config đã chốt từ validation.
   Dữ liệu test là file local riêng, KB đóng băng, không ghi."""
import json
from datetime import datetime
from scipy.stats import spearmanr, pearsonr
from eval import run_eval_once
from src.kb.knowledge_base import KnowledgeBase

BEST_MIN_IC    = 0.01
BEST_MAX_INNER = 3
MAX_OUTER_ITER = 4

TEST_START, TEST_END = "2024-09-01", "2025-04-01"

TEST_IDEAS = [
    "Tôi quan sát thấy giá tạo đỉnh mới nhưng khối lượng tại đỉnh thấp hơn khối lượng ở đỉnh trước đó. Đây có thể là phân kỳ giữa giá và dòng tiền. Hãy triển khai một tín hiệu giao dịch dựa trên quan sát này.",
    "Tôi quan sát thấy giá liên tục đóng cửa sát mức giá mở cửa (thân nến rất nhỏ) trong nhiều phiên. Thị trường có thể đang chờ một cú hích thông tin. Hãy triển khai một tín hiệu giao dịch dựa trên quan sát này.",
    "Tôi quan sát thấy khối lượng giao dịch trung bình 5 ngày đang vượt xa khối lượng trung bình 20 ngày trong khi giá vẫn đi ngang. Dòng tiền có thể đang âm thầm tích lũy trước khi giá phản ứng. Hãy triển khai một tín hiệu giao dịch dựa trên quan sát này.",
    "Tôi quan sát thấy tốc độ giảm giá đang chậm lại dần qua từng phiên dù xu hướng chính vẫn là giảm. Áp lực bán có thể đang cạn dần. Hãy triển khai một tín hiệu giao dịch dựa trên quan sát này.",
    "Tôi quan sát thấy giá thường xuyên chạm nhưng không đóng cửa vượt qua một ngưỡng giá cụ thể trong nhiều phiên liên tiếp. Ngưỡng kháng cự này có thể đang bị kiểm định nhiều lần trước khi vỡ. Hãy triển khai một tín hiệu giao dịch dựa trên quan sát này.",
    "Tôi quan sát thấy return của hôm nay có xu hướng lặp lại chiều với return của 1-2 ngày trước đó thay vì độc lập ngẫu nhiên. Đây có thể là dấu hiệu quán tính ngắn hạn trong chuỗi giá. Hãy triển khai một tín hiệu giao dịch dựa trên quan sát này.",
    "Tôi quan sát thấy phân phối return trong một cửa sổ gần đây bị lệch mạnh về một phía (nhiều phiên tăng nhẹ nhưng thỉnh thoảng có phiên giảm sâu, hoặc ngược lại). Sự bất cân xứng này có thể phản ánh rủi ro tiềm ẩn chưa thể hiện qua giá trung bình. Hãy triển khai một tín hiệu giao dịch dựa trên quan sát này.",
    "Tôi quan sát thấy trong một xu hướng tăng, giá có một nhịp điều chỉnh giảm ngắn rồi sau đó quay lại tăng và vượt đỉnh cũ trước điều chỉnh. Đây có thể là dấu hiệu xu hướng tăng còn nguyên vẹn sau khi hấp thụ áp lực chốt lời. Hãy triển khai một tín hiệu giao dịch dựa trên quan sát này.",
    "Tôi quan sát thấy có những phiên biên độ giá (high-low) rất rộng bất thường trong khi khối lượng giao dịch lại thấp hơn trung bình. Sự bất thường giữa biến động giá và thanh khoản có thể phản ánh thị trường mỏng hoặc thao túng ngắn hạn. Hãy triển khai một tín hiệu giao dịch dựa trên quan sát này.",
    "Tôi quan sát thấy số phiên đóng cửa cao hơn giá mở cửa trong một cửa sổ gần đây nhiều hơn hẳn so với mức biến động chung của cổ phiếu đó. Áp lực mua có thể đang mạnh hơn những gì độ biến động thông thường thể hiện. Hãy triển khai một tín hiệu giao dịch dựa trên quan sát này.",
]
LOG_PATH = "experiment_results/test_log.jsonl"

kb = KnowledgeBase()

all_judge_scores = []   # gộp từ mọi idea, mọi vòng — dùng để tính tương quan tổng thể
all_ic = []
all_sharpe = []

for idea in TEST_IDEAS:
    r = run_eval_once(idea, TEST_START, TEST_END, BEST_MIN_IC, BEST_MAX_INNER, MAX_OUTER_ITER, kb)

    history = r.history  # list các vòng outer: [{k, judge_score, ic, sharpe, ...}, ...]
    first_iter = history[0] if history else None
    last_iter  = history[-1] if history else None
    best_iter  = max(history, key=lambda h: h["ic"]) if history else None

    entry = {
        "timestamp": datetime.now().isoformat(),
        "idea": idea,
        "config": {"min_ic": BEST_MIN_IC, "max_inner": BEST_MAX_INNER},
        "final_metrics": r.best_metrics,          # metrics của alpha tốt nhất (đã lưu KB nếu qua threshold)
        "first_outer_iter": first_iter,             # MỚI
        "last_outer_iter": last_iter,                # MỚI
        "best_outer_iter": best_iter,                # MỚI — thường hữu ích hơn "last" để so sánh cải thiện
        "full_history": history,                     # MỚI — toàn bộ để vẽ đường cải thiện nếu cần
    }

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(entry)

    for h in history:
        all_judge_scores.append(h["judge_score"])
        all_ic.append(h["ic"])
        all_sharpe.append(h["sharpe"])

# ── Tương quan judge_score vs backtest metrics, gộp toàn bộ test set ──
if len(all_judge_scores) >= 5:   # cần tối thiểu vài điểm để correlation có ý nghĩa
    rho_ic, p_ic = spearmanr(all_judge_scores, all_ic)
    rho_sharpe, p_sharpe = spearmanr(all_judge_scores, all_sharpe)
    print(f"\n=== Tương quan Judge score vs Backtest (n={len(all_judge_scores)}) ===")
    print(f"Judge vs IC:      Spearman rho={rho_ic:.4f}, p={p_ic:.4f}")
    print(f"Judge vs Sharpe:  Spearman rho={rho_sharpe:.4f}, p={p_sharpe:.4f}")

    with open("experiment_results/judge_correlation.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_points": len(all_judge_scores),
            "judge_vs_ic":     {"spearman_rho": rho_ic, "p_value": p_ic},
            "judge_vs_sharpe": {"spearman_rho": rho_sharpe, "p_value": p_sharpe},
        }, f, indent=2)
else:
    print("Không đủ dữ liệu (< 5 điểm) để tính tương quan có ý nghĩa.")

print(f"\nĐã log vào {LOG_PATH}.")