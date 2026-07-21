"""
Chạy ablation TỪNG CONDITION MỘT — an toàn với rate limit, lưu ngay sau mỗi idea,
resume được nếu bị crash giữa chừng.

Cách dùng:
    python -m run_ablation --group 1 --condition no_loop
    python -m run_ablation --group 1 --condition only_inner_loop
    python -m run_ablation --group 1 --condition only_outer_loop
    python -m run_ablation --group 1 --condition full
    python -m run_ablation --group 2 --condition no_kb
    python -m run_ablation --group 2 --condition main_kb_only
    python -m run_ablation --group 2 --condition alpha101_only
    # "full" của group 2 giống hệt "full" của group 1 — không cần chạy lại,
    # plot_ablation.py sẽ tự dùng chung file ablation_g1_full.jsonl cho cả 2 group.

    # Nếu 1 idea bị lỗi (rate limit vượt retry, exception khác...), script KHÔNG dừng
    # toàn bộ — ghi lại lỗi, chuyển sang idea tiếp theo. Chạy lại cùng lệnh sẽ tự
    # bỏ qua các idea đã có kết quả (resume), chỉ chạy tiếp phần còn thiếu.
"""
import sys
import json
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import run_eval_once
from src.kb.knowledge_base import KnowledgeBase


DATA_START, DATA_END = "2024-01-01", "2024-08-31"
MAX_INNER, MAX_OUTER = 3, 4
MIN_IC_TO_SAVE = 0.01

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

GROUP1_CONFIGS = {
    "no_loop":         dict(max_inner_iter=1,         max_outer_iter=1,         use_main_kb=False, use_alpha101_kb=False),
    "only_inner_loop": dict(max_inner_iter=MAX_INNER, max_outer_iter=1,         use_main_kb=True,  use_alpha101_kb=True),
    "only_outer_loop": dict(max_inner_iter=1,         max_outer_iter=MAX_OUTER, use_main_kb=True,  use_alpha101_kb=True),
    "full":            dict(max_inner_iter=MAX_INNER, max_outer_iter=MAX_OUTER, use_main_kb=True,  use_alpha101_kb=True),
}
GROUP2_CONFIGS = {
    "no_kb":         dict(max_inner_iter=MAX_INNER, max_outer_iter=MAX_OUTER, use_main_kb=False, use_alpha101_kb=False),
    "main_kb_only":  dict(max_inner_iter=MAX_INNER, max_outer_iter=MAX_OUTER, use_main_kb=True,  use_alpha101_kb=False),
    "alpha101_only": dict(max_inner_iter=MAX_INNER, max_outer_iter=MAX_OUTER, use_main_kb=False, use_alpha101_kb=True),
    "full":          dict(max_inner_iter=MAX_INNER, max_outer_iter=MAX_OUTER, use_main_kb=True,  use_alpha101_kb=True),
}

OUT_DIR = Path("experiment_results")
OUT_DIR.mkdir(exist_ok=True)


def classify_verdict(judge_score_01: float, pass_score=6.0, min_acceptable=4.5) -> str:
    score_10 = judge_score_01 * 10
    if score_10 >= pass_score:
        return "GOOD"
    elif score_10 >= min_acceptable:
        return "WEAK"
    return "BAD"


def load_done_segments(out_path: Path) -> set:
    """Đọc file jsonl đã có, trả về set các segment index đã chạy xong (để resume)."""
    done = set()
    if not out_path.exists():
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(rec["segment"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def run_one_condition(group: int, cond_name: str, cfg: dict, ideas: list, retry_idea_on_fail: int = 1):
    out_path = OUT_DIR / f"ablation_g{group}_{cond_name}.jsonl"
    done_segments = load_done_segments(out_path)
    if done_segments:
        print(f"[resume] Đã có {len(done_segments)}/{len(ideas)} segment — chỉ chạy phần còn thiếu.")

    kb = KnowledgeBase()

    for seg, idea in enumerate(ideas):
        if seg in done_segments:
            print(f"  [{cond_name}] segment {seg} đã có sẵn — bỏ qua.")
            continue

        print(f"  [{cond_name}] segment {seg+1}/{len(ideas)} — đang chạy...")

        record = None
        last_err = None
        for attempt in range(retry_idea_on_fail + 1):
            try:
                r = run_eval_once(idea, DATA_START, DATA_END, MIN_IC_TO_SAVE, kb=kb, **cfg)
                m = r.best_metrics or {}
                history = r.history or []
                best_iter = max(history, key=lambda h: h["ic"]) if history else None
                judge_score_01 = best_iter["judge_score"] if best_iter else 0.0

                record = {
                    "segment":      seg,
                    "idea":         idea[:60],
                    "ic":           m.get("ic", 0.0),
                    "sharpe":       m.get("sharpe", 0.0),
                    "max_drawdown": m.get("max_drawdown", 0.0),
                    "valid_ratio":  m.get("valid_ratio", 0.0),
                    "judge_score":  round(judge_score_01 * 10, 2),
                    "verdict":      classify_verdict(judge_score_01),
                    "is_failed":    m.get("n_days", 0) == 0,
                    "error":        None,
                }
                break
            except Exception as e:
                last_err = str(e)
                print(f"    Lỗi ở segment {seg} (lần {attempt+1}): {last_err[:200]}")
                if attempt < retry_idea_on_fail:
                    print("    Nghỉ 30s trước khi thử lại...")
                    time.sleep(30)

        if record is None:
            # Vẫn ghi lại để không bị hỏi lại vô hạn, nhưng đánh dấu rõ là lỗi thật (khác is_failed backtest)
            record = {
                "segment": seg, "idea": idea[:60],
                "ic": None, "sharpe": None, "max_drawdown": None, "valid_ratio": None,
                "judge_score": None, "verdict": None, "is_failed": True,
                "error": last_err,
            }
            print(f"    !! Segment {seg} lỗi hẳn sau {retry_idea_on_fail+1} lần thử — ghi lại error, "
                  f"chạy lại lệnh này sau để retry riêng segment này.")

        # Ghi ngay lập tức — KHÔNG đợi hết condition mới ghi
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nXong condition '{cond_name}'. Kết quả: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=int, choices=[1, 2], required=True)
    parser.add_argument("--condition", type=str, required=True,
                         help="Tên condition, ví dụ: no_loop, only_inner_loop, only_outer_loop, "
                              "full, no_kb, main_kb_only, alpha101_only")
    args = parser.parse_args()

    configs = GROUP1_CONFIGS if args.group == 1 else GROUP2_CONFIGS
    if args.condition not in configs:
        raise ValueError(f"Condition '{args.condition}' không tồn tại trong group {args.group}. "
                          f"Các lựa chọn: {list(configs.keys())}")

    run_one_condition(args.group, args.condition, configs[args.condition], VALIDATION_IDEAS)