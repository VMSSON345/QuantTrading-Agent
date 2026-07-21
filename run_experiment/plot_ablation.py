"""
Đọc kết quả đã chạy (từ run_ablation.py) và vẽ chart — chạy SAU KHI đã chạy xong
đủ các condition cần thiết bằng run_ablation.py.

Cách dùng:
    python -m scripts.plot_ablation --group 1
    python -m scripts.plot_ablation --group 2
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

OUT_DIR = Path("experiment_results")

GROUP1_ORDER = ["no_loop", "only_inner_loop", "only_outer_loop", "full"]
GROUP2_ORDER = ["no_kb", "main_kb_only", "alpha101_only", "full"]

COLORS = {
    "no_loop":         "tab:red",
    "only_inner_loop": "tab:orange",
    "only_outer_loop": "tab:purple",
    "no_kb":           "tab:brown",
    "main_kb_only":    "tab:cyan",
    "alpha101_only":   "tab:green",
    "full":            "tab:blue",
}
MARKERS = {
    "no_loop":         "o",
    "only_inner_loop": "s",
    "only_outer_loop": "^",
    "no_kb":           "v",
    "main_kb_only":    "P",
    "alpha101_only":   "X",
    "full":            "D",
}
LABELS = {
    "no_loop":         "No loop",
    "only_inner_loop": "Inner only",
    "only_outer_loop": "Outer only",
    "no_kb":           "No KB",
    "main_kb_only":    "Main KB only",
    "alpha101_only":   "Alpha101 only",
    "full":            "Full",
}


def load_condition(group: int, cond_name: str) -> list:
    """Group 2 'full' dùng chung file với group 1 'full' nếu file group 2 chưa tồn tại."""
    path = OUT_DIR / f"ablation_g{group}_{cond_name}.jsonl"
    if not path.exists() and cond_name == "full":
        alt_group = 1 if group == 2 else 2
        alt_path = OUT_DIR / f"ablation_g{alt_group}_full.jsonl"
        if alt_path.exists():
            print(f"[info] Dùng chung file 'full' từ group {alt_group}: {alt_path}")
            path = alt_path
        else:
            raise FileNotFoundError(f"Chưa có kết quả cho 'full' — chạy: "
                                     f"python -m scripts.run_ablation --group {group} --condition full")
    if not path.exists():
        raise FileNotFoundError(f"Chưa có kết quả: {path} — chạy: "
                                 f"python -m scripts.run_ablation --group {group} --condition {cond_name}")

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r["segment"])

    n_error = sum(1 for r in records if r.get("error"))
    if n_error:
        print(f"[cảnh báo] '{cond_name}': {n_error} segment vẫn lỗi (chưa retry thành công). "
              f"Chạy lại: python -m scripts.run_ablation --group {group} --condition {cond_name}")

    return [r for r in records if r.get("error") is None]  # bỏ segment còn lỗi thật khi vẽ


def plot_group(group_title: str, results: dict, save_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    conditions = list(results.keys())

    ax = axes[0, 0]
    for cond in conditions:
        segs = [r["segment"] for r in results[cond]]
        vals = [r["ic"] for r in results[cond]]
        ax.plot(segs, vals, marker=MARKERS.get(cond, "o"), color=COLORS.get(cond), label=LABELS.get(cond, cond))
    ax.set_title("IC score"); ax.set_xlabel("Segment"); ax.set_ylabel("IC"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    for cond in conditions:
        segs = [r["segment"] for r in results[cond]]
        vals = [r["sharpe"] for r in results[cond]]
        ax.plot(segs, vals, marker=MARKERS.get(cond, "o"), color=COLORS.get(cond), label=LABELS.get(cond, cond))
    ax.set_title("Sharpe Ratio"); ax.set_xlabel("Segment"); ax.set_ylabel("Sharpe"); ax.legend(fontsize=8)

    ax = axes[1, 0]
    for cond in conditions:
        segs = [r["segment"] for r in results[cond]]
        vals = [r["max_drawdown"] for r in results[cond]]
        ax.plot(segs, vals, marker=MARKERS.get(cond, "o"), color=COLORS.get(cond), label=LABELS.get(cond, cond))
    ax.set_title("Max Drawdown"); ax.set_xlabel("Segment"); ax.set_ylabel("Max Drawdown"); ax.legend(fontsize=8)

    ax = axes[1, 1]
    for cond in conditions:
        segs = [r["segment"] for r in results[cond]]
        vals = [r["valid_ratio"] for r in results[cond]]
        ax.plot(segs, vals, marker=MARKERS.get(cond, "o"), color=COLORS.get(cond), label=LABELS.get(cond, cond))
    ax.set_title("Valid Ratio"); ax.set_xlabel("Segment"); ax.set_ylabel("Valid Ratio"); ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Đã lưu chart: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=int, choices=[1, 2], required=True)
    args = parser.parse_args()

    order = GROUP1_ORDER if args.group == 1 else GROUP2_ORDER
    results = {cond: load_condition(args.group, cond) for cond in order}

    title = ("Ablation Nhóm 1: No-loop vs Inner-only vs Outer-only vs Full" if args.group == 1
              else "Ablation Nhóm 2: No-KB vs Main-KB-only vs Alpha101-only vs Full")
    plot_group(title, results, OUT_DIR / f"ablation_group{args.group}.png")

    print(f"\n=== Tóm tắt Group {args.group} ===")
    for cond, recs in results.items():
        valid = [r for r in recs if not r["is_failed"]]
        fail_rate = 1 - len(valid) / len(recs) if recs else None
        avg_ic = sum(r["ic"] for r in valid) / len(valid) if valid else None
        avg_sharpe = sum(r["sharpe"] for r in valid) / len(valid) if valid else None
        print(f"[{cond}] n={len(recs)} fail_rate={fail_rate:.1%} avg_IC={avg_ic} avg_Sharpe={avg_sharpe}")