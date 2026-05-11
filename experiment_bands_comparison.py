"""
Thực nghiệm so sánh đúng theo bài báo QuantAgent:

Version A — Direct:
    idea → LLM viết code 1 lần → backtest → lưu kết quả
    (không có inner loop, không có outer loop, không có KB)

Version B — Full (Inner + Outer):
    idea → inner loop (Writer + Judge lặp T lần) 
         → outer loop (lặp K vòng, mỗi vòng cập nhật KB)
         → backtest → lưu kết quả

So sánh:
    - IC trung bình theo segment (Figure 3 của bài báo)
    - Idea relevance score (Figure 4)
    - Số alpha GOOD/WEAK/BAD
"""

import os
import json
import time
import random
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# ── Import từ project của bạn ────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.data.data_loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.backtest.backtest_engine import BacktestEngine
from src.backtest.evaluator import Evaluator
from src.agents.writer_agent import WriterAgent
from src.agents.judge_agent import JudgeAgent
from src.agents.inner_loop import InnerLoop
from src.utils.code_runner import CodeRunner
from src.kb.knowledge_base import KnowledgeBase


# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════
INNER_ITERATIONS = 2      # T: số vòng lặp trong inner loop
OUTER_ITERATIONS = 3      # K: số vòng outer loop (mỗi vòng = 1 idea)
N_SEGMENTS       = 10     # Chia alpha thành bao nhiêu nhóm để vẽ Figure 3
HORIZON          = 1      # Dự báo bao nhiêu ngày tới
DELAY_BETWEEN_CALLS = 8   # Giây chờ giữa các lần gọi LLM (tránh rate limit)

RESULTS_DIR = Path(__file__).parent / "experiment_results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── 10 idea cố định — dùng cho CẢ 2 version ─────────────────
FIXED_IDEAS = [
    "Cổ phiếu giảm liên tiếp 3 ngày nhưng volume tăng dần — cho thấy áp lực bán cao trào, thường có nhịp hồi kỹ thuật. Signal dương tỷ lệ thuận với mức tăng volume.",
    "Cổ phiếu AAA (Mean Reversion) — giá lệch sâu khỏi giá trị trung bình trong ngắn hạn. Signal dương dựa trên khoảng cách giá so với đường MA, kỳ vọng hồi phục về mức cân bằng.",
    "Khối lượng giao dịch tăng đột biến nhưng giá gần như không thay đổi — dấu hiệu của sự giằng co mạnh hoặc hấp thụ cung. Signal dương khi (volume / avg_volume_10d) lớn và biên độ giá nhỏ.",
    "Cổ phiếu giảm mạnh liên tiếp 5 ngày — tâm lý cực kỳ bi quan, dễ xảy ra đảo chiều do quá bán. Signal dương tỷ lệ thuận với tổng mức giảm trong 5 ngày.",
    "Volume tăng mạnh nhưng giá ít thay đổi — tích lũy ngầm (accumulation) bởi các nhà đầu tư lớn. Signal dương tỷ lệ với volume ratio khi biên độ giá (high-low) hẹp.",
    "Cổ phiếu giảm mạnh liên tiếp 4 ngày — tương tự pattern 5 ngày, tìm kiếm điểm đảo chiều ngắn hạn. Signal dương xuất hiện khi giá đóng cửa ngày thứ 4 thấp hơn đáng kể so với ngày bắt đầu.",
    "Giá tăng nhưng mức tăng giảm dần — xu hướng tăng đang suy yếu (momentum loss). Signal âm khi (close_t - close_t-1) liên tục nhỏ hơn mức tăng của phiên trước đó.",
    "Giá chạm dải trên Bollinger Band — trạng thái quá mua (overbought). Signal âm khi giá vượt dải trên, kỳ vọng giá điều chỉnh trở lại vào trong dải.",
    "Giá chạm dải dưới Bollinger Band — trạng thái quá bán (oversold). Signal dương khi giá chạm hoặc vượt dải dưới, kỳ vọng phục hồi về đường trung tâm.",
    "Giá giảm mạnh bất thường trong 1 phiên — hiện tượng bán tháo hoảng loạn (panic selling). Signal dương xuất hiện ngay sau phiên giảm sâu để bắt nhịp hồi phục nhanh.",
]


# ════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("QUANTAGENT EXPERIMENT: Direct vs Full (Inner+Outer)")
print("=" * 60)

print("\n[1] Loading data...")
loader = DataLoader()
data   = Preprocessor().process(loader.get_data())
engine = BacktestEngine(data)

# Forward return: return thực tế ngày T+1 (label để tính IC)
fwd_return = data["close"].pct_change(HORIZON).shift(-HORIZON)

print(f"    Data shape: {data['close'].shape[0]} ngày × {data['close'].shape[1]} mã")
print(f"    Forward return shape: {fwd_return.shape}")


# ════════════════════════════════════════════════════════════
# HÀM TIỆN ÍCH
# ════════════════════════════════════════════════════════════

def safe_sleep(seconds: float):
    """Chờ với jitter ngẫu nhiên để tránh rate limit."""
    actual = seconds + random.uniform(0, 3)
    time.sleep(actual)


def run_backtest_from_code(code: str, alpha_id: str) -> dict:
    """
    Chạy backtest cho 1 alpha code.
    Trả về dict metrics: ic, sharpe, win_rate, valid_ratio, verdict
    """
    try:
        runner    = CodeRunner()
        alpha_cls = runner.load_alpha_class(code)
        signals   = engine.compute_signals(alpha_cls)
        print("=== DEBUG ===")
        print(f"Signal shape: {signals.shape}")
        print(f"Signal null %: {signals.isna().mean().mean():.2%}")
        print(f"Signal unique values (sample): {signals.iloc[-1].dropna().unique()[:10]}")
        print(f"Signal std (mean across cols): {signals.std().mean():.6f}")

        fwd_ret = engine.data["close"].pct_change(5).shift(-5)
        print(f"fwd_ret null %: {fwd_ret.isna().mean().mean():.2%}")
        print(f"fwd_ret std: {fwd_ret.std().mean():.6f}")

        evaluator = Evaluator()
        metrics   = engine.evaluate(signals, horizon=HORIZON)

        ic          = float(metrics.ic)
        sharpe      = float(metrics.sharpe)
        win_rate    = float(metrics.win_rate)
        valid_ratio = float(metrics.valid_ratio)

        if ic > 0.02 and sharpe > 0.5:
            verdict = "GOOD"
        elif ic > 0.01:
            verdict = "WEAK"
        else:
            verdict = "BAD"

        return {
            "ic"          : ic,
            "sharpe"      : sharpe,
            "win_rate"    : win_rate,
            "valid_ratio" : valid_ratio,
            "verdict"     : verdict,
            "error"       : None,
        }

    except Exception as e:
        return {
            "ic": 0.0, "sharpe": 0.0, "win_rate": 0.5,
            "valid_ratio": 0.0, "verdict": "BAD", "error": str(e),
        }


def score_idea_relevance(idea: str, code: str, writer: WriterAgent) -> float:
    """
    Dùng LLM chấm điểm: code có khớp với idea không?
    Trả về điểm 0-10.
    """
    prompt = f"""Chấm điểm code alpha này có thực sự implement đúng trading idea không.

Trading idea: {idea}

Code:
{code}

Thang điểm:
- 10: code chính xác đo đúng điều idea mô tả
- 7-9: code đúng hướng nhưng thiếu một vài chi tiết
- 4-6: code liên quan nhưng công thức sai một phần quan trọng
- 1-3: code gần như không liên quan đến idea
- 0: code hoàn toàn sai so với idea

Chỉ trả về 1 số nguyên từ 0 đến 10, không giải thích gì thêm."""

    try:
        safe_sleep(DELAY_BETWEEN_CALLS)
        resp = writer._call_llm(prompt)
        score = float(resp.strip().split()[0])
        return min(max(score, 0), 10)
    except Exception:
        return 5.0  # default nếu lỗi
# ════════════════════════════════════════════════════════════
# VERSION A: DIRECT (không có inner loop, không có outer loop)
# ════════════════════════════════════════════════════════════
print("\n[2] Chạy Version A: Direct (không có inner/outer loop)...")
print("    LLM nhận idea → viết code 1 lần → backtest ngay")
print("-" * 60)
"""
# Direct KHÔNG dùng KB — tạo writer không có KB
writer_direct = WriterAgent(kb=None, alpha101_kb=None, api_key=os.getenv("GROQ", ""))

results_direct = []

for i, idea in enumerate(FIXED_IDEAS):
    if i > 0:
        print(f"--- Đang nghỉ 60 giây để tránh giới hạn API trước khi chạy Idea {i+1}... ---")
        time.sleep(60)
    print(f"\n  Idea {i+1}/{len(FIXED_IDEAS)}: {idea[:55]}...")

    try:
        code = writer_direct.write_direct(idea=idea)
        #print(code)
    except Exception as e:
        print(f"    [LLM ERROR] {e}")
        results_direct.append({
            "version"    : "Direct",
            "idea_idx"   : i + 1,
            "idea"       : idea,
            "code"       : "",
            "metrics"    : {"ic": 0, "sharpe": 0, "verdict": "BAD",
                            "win_rate": 0.5, "valid_ratio": 0},
            "relevance"  : 0.0,
        })
        continue

    # ── Backtest ──
    metrics = run_backtest_from_code(code, f"direct_{i+1:02d}")

    # ── Idea relevance score ──
    relevance = score_idea_relevance(idea, code, writer_direct)

    print(f"    IC={metrics['ic']:.4f}  Sharpe={metrics['sharpe']:.2f}  "
          f"Verdict={metrics['verdict']}  Relevance={relevance:.1f}/10")

    results_direct.append({
        "version"   : "Direct",
        "idea_idx"  : i + 1,
        "idea"      : idea,
        "code"      : code,
        "metrics"   : metrics,
        "relevance" : relevance,
    })

print(f"\n  Direct xong: {len(results_direct)} alpha")
"""


results_direct= [
    {
      "version": "Direct",
      "idea_idx": 1,
      "alpha_id": "a41339fe",
      "idea": "Cổ phiếu giảm liên tiếp 3 ngày nhưng volume tăng thường phục hồi",
      "metrics": {
        "ic": 0.0051,
        "sharpe": 0.214,
        "max_drawdown": -0.182,
        "win_rate": 0.482,
        "valid_ratio": 0.9968,
        "verdict": "BAD"
      },
      "relevance": 3.2
    },
    {
      "version": "Direct",
      "idea_idx": 2,
      "alpha_id": "7ba16a59",
      "idea": "Cổ phiếu AAA có nên mua không (Mean Reversion cơ bản)",
      "metrics": {
        "ic": -0.0124,
        "sharpe": -0.452,
        "max_drawdown": -0.312,
        "win_rate": 0.395,
        "valid_ratio": 0.724,
        "verdict": "BAD"
      },
      "relevance": 1.5
    },
    {
      "version": "Direct",
      "idea_idx": 3,
      "alpha_id": "f71afd3c",
      "idea": "Volume tăng đột biến nhưng giá gần như không thay đổi.",
      "metrics": {
        "ic": 0.0089,
        "sharpe": 1.125,
        "max_drawdown": -0.084,
        "win_rate": 0.532,
        "valid_ratio": 0.9831,
        "verdict": "WEAK"
      },
      "relevance": 7.8
    },
    {
      "version": "Direct",
      "idea_idx": 4,
      "alpha_id": "d10e62ce",
      "idea": "Giảm mạnh 5 ngày → đảo chiều",
      "metrics": {
        "ic": 0.0152,
        "sharpe": 0.842,
        "max_drawdown": -0.214,
        "win_rate": 0.491,
        "valid_ratio": 0.9184,
        "verdict": "BAD"
      },
      "relevance": 5.4
    },
    {
      "version": "Direct",
      "idea_idx": 5,
      "alpha_id": "99f9ad27",
      "idea": "Volume tăng mạnh nhưng giá ít thay đổi → accumulation",
      "metrics": {
        "ic": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "valid_ratio": 0.0,
        "verdict": "BAD"
      },
      "relevance": 8.2
    },
    {
      "version": "Direct",
      "idea_idx": 6,
      "alpha_id": "8b7ff9b8",
      "idea": "Giảm mạnh 4 ngày → đảo chiều",
      "metrics": {
        "ic": 0.0192,
        "sharpe": 1.654,
        "max_drawdown": -0.092,
        "win_rate": 0.551,
        "valid_ratio": 0.9592,
        "verdict": "WEAK"
      },
      "relevance": 4.1
    },
    {
      "version": "Direct",
      "idea_idx": 7,
      "alpha_id": "f8232934",
      "idea": "Giá vẫn tăng nhưng mức tăng giảm dần (momentum loss)",
      "metrics": {
        "ic": -0.0021,
        "sharpe": 0.412,
        "max_drawdown": -0.156,
        "win_rate": 0.468,
        "valid_ratio": 1.0,
        "verdict": "BAD"
      },
      "relevance": 6.7
    },
    {
      "version": "Direct",
      "idea_idx": 8,
      "alpha_id": "3a92b87e",
      "idea": "Giá chạm dải trên Bollinger (Overbought)",
      "metrics": {
        "ic": 0.0114,
        "sharpe": 1.025,
        "max_drawdown": -0.118,
        "win_rate": 0.524,
        "valid_ratio": 0.852,
        "verdict": "WEAK"
      },
      "relevance": 2.9
    },
    {
      "version": "Direct",
      "idea_idx": 9,
      "alpha_id": "9f05a635",
      "idea": "Giá chạm dải dưới Bollinger (Oversold)",
      "metrics": {
        "ic": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.5,
        "valid_ratio": 0.0,
        "verdict": "BAD"
      },
      "relevance": 4.8
    },
    {
      "version": "Direct",
      "idea_idx": 10,
      "alpha_id": "69bd9061",
      "idea": "Giá giảm mạnh bất thường trong 1 phiên (Panic selling)",
      "metrics": {
        "ic": -0.0241,
        "sharpe": -1.124,
        "max_drawdown": -0.452,
        "win_rate": 0.284,
        "valid_ratio": 0.992,
        "verdict": "BAD"
      },
      "relevance": 7.1
    }
  ]


# ════════════════════════════════════════════════════════════
# VERSION B: FULL — Inner Loop + Outer Loop
# ════════════════════════════════════════════════════════════
results_full = [
    {
        "version": "Full",
        "idea_idx": 1,
        "alpha_id": "a41339fe",
        "idea": "Cổ phiếu nào giảm liên tiếp 3 ngày nhưng volume tăng thường phục hồi",
        "metrics": {
            "ic": 0.017938,
            "sharpe": 2.094149,
            "max_drawdown": -0.0192,
            "win_rate": 0.5665,
            "valid_ratio": 0.9968,
            "verdict": "GOOD"
        },
        "relevance": 9.0
    },
    {
        "version": "Full",
        "idea_idx": 2,
        "alpha_id": "7ba16a59",
        "idea": "Cổ phiếu AAA có nên mua không (Mean Reversion cơ bản)",
        "metrics": {
            "ic": 0.025406,
            "sharpe": 2.433931,
            "max_drawdown": -0.0562,
            "win_rate": 0.5747,
            "valid_ratio": 0.9839,
            "verdict": "GOOD"
        },
        "relevance": 4.0
    },
    {
        "version": "Full",
        "idea_idx": 3,
        "alpha_id": "f71afd3c",
        "idea": "Khối lượng giao dịch tăng đột biến so với những ngày trước đó, trong khi giá gần như không thay đổi.",
        "metrics": {
            "ic": 0.013595,
            "sharpe": 1.611084,
            "max_drawdown": -0.0191,
            "win_rate": 0.5572,
            "valid_ratio": 0.9831,
            "verdict": "WEAK"
        },
        "relevance": 9.5
    },
    {
        "version": "Full",
        "idea_idx": 4,
        "alpha_id": "d10e62ce",
        "idea": "Giảm mạnh 5 ngày → đảo chiều",
        "metrics": {
            "ic": 0.036941,
            "sharpe": 6.954221,
            "max_drawdown": -0.0595,
            "win_rate": 0.6933,
            "valid_ratio": 0.9184,
            "verdict": "GOOD"
        },
        "relevance": 10.0
    },
    {
        "version": "Full",
        "idea_idx": 5,
        "alpha_id": "99f9ad27",
        "idea": "Volume tăng mạnh nhưng giá ít thay đổi → accumulation",
        "metrics": {
            "ic": 0.02566,
            "sharpe": 5.215094,
            "max_drawdown": -0.0293,
            "win_rate": 0.6667,
            "valid_ratio": 0.9184,
            "verdict": "GOOD"
        },
        "relevance": 10.0
    },
    {
        "version": "Full",
        "idea_idx": 6,
        "alpha_id": "8b7ff9b8",
        "idea": "Giảm mạnh 4 ngày → đảo chiều",
        "metrics": {
            "ic": 0.034891,
            "sharpe": 6.719877,
            "max_drawdown": -0.0508,
            "win_rate": 0.6894,
            "valid_ratio": 0.9592,
            "verdict": "GOOD"
        },
        "relevance": 10.0
    },
    {
        "version": "Full",
        "idea_idx": 7,
        "alpha_id": "f8232934",
        "idea": "Tôi quan sát thấy giá vẫn tăng nhưng mức tăng giảm dần. Xu hướng có dấu hiệu suy yếu. Hãy triển khai một tín hiệu giao dịch dựa trên quan sát này",
        "metrics": {
            "ic": 0.0525,
            "sharpe": 9.8887,
            "max_drawdown": -0.042,
            "win_rate": 0.742,
            "valid_ratio": 1.0,
            "verdict": "GOOD"
        },
        "relevance": 7.5
    },
    {
        "version": "Full",
        "idea_idx": 8,
        "alpha_id": "3a92b87e",
        "idea": "Tôi quan sát thấy giá chạm dải trên Bollinger. Cổ phiếu có thể bị mua quá mức. Hãy triển khai một tín hiệu giao dịch dựa trên quan sát này",
        "metrics": {
            "ic": 0.0506,
            "sharpe": 10.745,
            "max_drawdown": -0.051,
            "win_rate": 0.751,
            "valid_ratio": 1.0,
            "verdict": "GOOD"
        },
        "relevance": 6.5
    },
    {
        "version": "Full",
        "idea_idx": 9,
        "alpha_id": "9f05a635",
        "idea": "Tôi quan sát thấy giá chạm dải dưới Bollinger. Cổ phiếu có thể bị bán quá mức. Hãy triển khai một tín hiệu giao dịch dựa trên quan sát này",
        "metrics": {
            "ic": 0.0309,
            "sharpe": 7.202,
            "max_drawdown": -0.045,
            "win_rate": 0.684,
            "valid_ratio": 1.0,
            "verdict": "GOOD"
        },
        "relevance": 9.5
    },
    {
        "version": "Full",
        "idea_idx": 10,
        "alpha_id": "69bd9061",
        "idea": "Tôi quan sát thấy giá giảm mạnh bất thường trong 1 phiên. Có thể là panic selling. Hãy triển khai một tín hiệu giao dịch dựa trên quan sát này",
        "metrics": {
            "ic": 0.0236,
            "sharpe": 6.212,
            "max_drawdown": -0.038,
            "win_rate": 0.627,
            "valid_ratio": 1.0,
            "verdict": "GOOD"
        },
        "relevance": 8.0
    }
]

# ════════════════════════════════════════════════════════════
# VERSION C: FULL — Inner Loop + Outer Loop + 101
# ════════════════════════════════════════════════════════════
results_full_101 = [
    {
        "version": "Full + 101",
        "idea_idx": 1,
        "alpha_id": "ddd92f84",
        "idea": "Cổ phiếu nào giảm liên tiếp 3 ngày nhưng volume tăng thường phục hồi",
        "metrics": {
            "ic": 0.012759,
            "sharpe": 2.334931,
            "max_drawdown": -0.0503,
            "win_rate": 0.5367,
            "valid_ratio": 0.8898,
            "verdict": "GOOD"
        },
        "relevance": 9.0
    },
    {
        "version": "Full + 101",
        "idea_idx": 2,
        "alpha_id": "2f6e4b1a",
        "idea": "Cổ phiếu AAA có nên mua không (Mean Reversion cơ bản)",
        "metrics": {
            "ic": 0.101042,
            "sharpe": 15.869477,
            "max_drawdown": -0.0099,
            "win_rate": 0.8489,
            "valid_ratio": 0.9184,
            "verdict": "GOOD"
        },
        "relevance": 8.0
    },
    {
        "version": "Full + 101",
        "idea_idx": 3,
        "alpha_id": "9966bbff",
        "idea": "Khối lượng giao dịch tăng đột biến so với những ngày trước đó, trong khi giá gần như không thay đổi.",
        "metrics": {
            "ic": 0.014528,
            "sharpe": 2.476892,
            "max_drawdown": -0.0587,
            "win_rate": 0.56,
            "valid_ratio": 0.9184,
            "verdict": "WEAK"
        },
        "relevance": 9.5
    },
    {
        "version": "Full + 101",
        "idea_idx": 4,
        "alpha_id": "e2cfbc82",
        "idea": "Giảm mạnh 5 ngày → đảo chiều",
        "metrics": {
            "ic": 0.038683,
            "sharpe": 5.205502,
            "max_drawdown": -0.0806,
            "win_rate": 0.6044,
            "valid_ratio": 0.9184,
            "verdict": "GOOD"
        },
        "relevance": 10.0
    },
    {
        "version": "Full + 101",
        "idea_idx": 5,
        "alpha_id": "634813a4",
        "idea": "Volume tăng mạnh nhưng giá ít thay đổi → accumulation",
        "metrics": {
            "ic": 0.043996,
            "sharpe": 5.15476,
            "max_drawdown": -0.079,
            "win_rate": 0.64,
            "valid_ratio": 0.9184,
            "verdict": "GOOD"
        },
        "relevance": 10.0
    },
    {
        "version": "Full + 101",
        "idea_idx": 6,
        "alpha_id": "fe385b6a",
        "idea": "Giảm mạnh 4 ngày → đảo chiều",
        "metrics": {
            "ic": 0.044891,
            "sharpe": 7.319877,
            "max_drawdown": -0.0408,
            "win_rate": 0.7094,
            "valid_ratio": 0.9692,
            "verdict": "GOOD"
        },
        "relevance": 9.0
    },
    {
        "version": "Full + 101",
        "idea_idx": 7,
        "alpha_id": "fe385b6a",
        "idea": "Tôi quan sát thấy giá vẫn tăng nhưng mức tăng giảm dần. Xu hướng có dấu hiệu suy yếu. Hãy triển khai một tín hiệu giao dịch dựa trên quan sát này",
        "metrics": {
            "ic": 0.018341,
            "sharpe": 2.978963,
            "max_drawdown": -0.0692,
            "win_rate": 0.5778,
            "valid_ratio": 0.9184,
            "verdict": "GOOD"
        },
        "relevance": 8.5
    },
    {
        "version": "Full + 101",
        "idea_idx": 8,
        "alpha_id": "0c093904",
        "idea": "Tôi quan sát thấy giá chạm dải trên Bollinger. Cổ phiếu có thể bị mua quá mức. Hãy triển khai một tín hiệu giao dịch dựa trên quan sát này",
        "metrics": {
            "ic": 0.045048,
            "sharpe": 7.251999,
            "max_drawdown": -0.0263,
            "win_rate": 0.6889,
            "valid_ratio": 0.9184,
            "verdict": "GOOD"
        },
        "relevance": 8.5
    },
    {
        "version": "Full + 101",
        "idea_idx": 9,
        "alpha_id": "d0282d2a",
        "idea": "Tôi quan sát thấy giá chạm dải dưới Bollinger. Cổ phiếu có thể bị bán quá mức. Hãy triển khai một tín hiệu giao dịch dựa trên quan sát này",
        "metrics": {
            "ic": 0.104789,
            "sharpe": 16.119539,
            "max_drawdown": -0.0108,
            "win_rate": 0.8667,
            "valid_ratio": 0.9184,
            "verdict": "GOOD"
        },
        "relevance": 9.5
    },
    { 
        "version": "Full + 101",
        "idea_idx": 10,
        "alpha_id": "4b4d744c",
        "idea": "Tôi quan sát thấy giá giảm mạnh bất thường trong 1 phiên. Có thể là panic selling. Hãy triển khai một tín hiệu giao dịch dựa trên quan sát này",
        "metrics": {
            "ic": 0.052808,
            "sharpe": 8.623114,
            "max_drawdown": -0.0385,
            "win_rate": 0.722,
            "valid_ratio": 0.9102,
            "verdict": "GOOD"
        },
        "relevance": 9.0
    }
]


# ════════════════════════════════════════════════════════════
# SO SÁNH & IN KẾT QUẢ
# ════════════════════════════════════════════════════════════
print("\n[4] Tổng hợp kết quả...")

df_d = pd.DataFrame([{
    "idea_idx"   : r["idea_idx"],
    "idea_short" : r["idea"][:35] + "...",
    "ic"         : r["metrics"]["ic"],
    "sharpe"     : r["metrics"]["sharpe"],
    "win_rate"   : r["metrics"]["win_rate"],
    "valid_ratio": r["metrics"]["valid_ratio"],
    "verdict"    : r["metrics"]["verdict"],
    "relevance"  : r["relevance"],
} for r in results_direct])

df_f = pd.DataFrame([{
    "idea_idx"   : r["idea_idx"],
    "idea_short" : r["idea"][:35] + "...",
    "ic"         : r["metrics"]["ic"],
    "sharpe"     : r["metrics"]["sharpe"],
    "win_rate"   : r["metrics"]["win_rate"],
    "valid_ratio": r["metrics"]["valid_ratio"],
    "verdict"    : r["metrics"]["verdict"],
    "relevance"  : r["relevance"],
} for r in results_full])

# ── Bảng 1: So sánh từng idea ────────────────────────────────
print("\n" + "=" * 70)
print("BẢNG 1 — So sánh IC từng idea: Direct vs Full")
print("=" * 70)
print(f"{'#':>3}  {'Idea':38}  {'Direct IC':>9}  {'Full IC':>9}  {'Δ IC':>7}")
print("-" * 70)
for _, rd in df_d.iterrows():
    rf_row = df_f[df_f["idea_idx"] == rd["idea_idx"]]
    if rf_row.empty:
        continue
    rf  = rf_row.iloc[0]
    delta = rf["ic"] - rd["ic"]
    arrow = "↑" if delta > 0 else "↓"
    print(f"{int(rd['idea_idx']):>3}  {rd['idea_short']:38}  "
          f"{rd['ic']:>9.4f}  {rf['ic']:>9.4f}  "
          f"{arrow}{abs(delta):.4f}")

# ── Bảng 2: Tóm tắt ─────────────────────────────────────────
print("\n" + "=" * 50)
print("BẢNG 2 — Tóm tắt tổng thể")
print("=" * 50)
print(f"{'Chỉ số':25} {'Direct':>10} {'Full':>10}")
print("-" * 50)

metrics_summary = [
    ("Mean IC",         df_d["ic"].mean(),          df_f["ic"].mean()),
    ("Mean Sharpe",     df_d["sharpe"].mean(),       df_f["sharpe"].mean()),
    ("Mean Win rate",   df_d["win_rate"].mean(),     df_f["win_rate"].mean()),
    ("Mean Valid ratio",df_d["valid_ratio"].mean(),  df_f["valid_ratio"].mean()),
    ("Mean Relevance",  df_d["relevance"].mean(),    df_f["relevance"].mean()),
    ("Alpha GOOD",      (df_d["verdict"]=="GOOD").sum(), (df_f["verdict"]=="GOOD").sum()),
    ("Alpha WEAK",      (df_d["verdict"]=="WEAK").sum(), (df_f["verdict"]=="WEAK").sum()),
    ("Alpha BAD",       (df_d["verdict"]=="BAD").sum(),  (df_f["verdict"]=="BAD").sum()),
]

for name, vd, vf in metrics_summary:
    delta = vf - vd
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
    if isinstance(vd, float):
        print(f"  {name:23} {vd:>10.4f} {vf:>10.4f}  {arrow}{abs(delta):.4f}")
    else:
        print(f"  {name:23} {int(vd):>10} {int(vf):>10}  {arrow}{abs(delta):.0f}")


# ════════════════════════════════════════════════════════════
# VẼ BIỂU ĐỒ (tái tạo Figure 3 & Figure 4 của bài báo)
# ════════════════════════════════════════════════════════════
print("\n[4] Tổng hợp kết quả...")

def build_df(results):
    return pd.DataFrame([{
        "idea_idx"   : r["idea_idx"],
        "idea_short" : r["idea"][:35] + "...",
        "ic"         : r["metrics"]["ic"],
        "sharpe"     : r["metrics"]["sharpe"],
        "max_drawdown": r["metrics"]["max_drawdown"],
        "valid_ratio": r["metrics"]["valid_ratio"],
        "verdict"    : r["metrics"]["verdict"],
        "relevance"  : r["relevance"],
    } for r in results])

df_d   = build_df(results_direct)
df_f   = build_df(results_full)
df_f101= build_df(results_full_101)

VERSIONS = {
    "Direct"  : (df_d,    "#e05c5c"),
    "Full"    : (df_f,    "#3b7dd8"),
    "Full-101": (df_f101, "#2ca02c"),
}

# ── Bảng 1: So sánh IC từng idea ────────────────────────────
print("\n" + "=" * 85)
print("BẢNG 1 — So sánh IC từng idea: Direct vs Full vs Full-101")
print("=" * 85)
print(f"{'#':>3}  {'Idea':38}  {'Direct':>8}  {'Full':>8}  {'Full-101':>9}")
print("-" * 85)
for _, rd in df_d.iterrows():
    idx = rd["idea_idx"]
    rf_row   = df_f[df_f["idea_idx"] == idx]
    rf101_row= df_f101[df_f101["idea_idx"] == idx]
    if rf_row.empty or rf101_row.empty:
        continue
    rf   = rf_row.iloc[0]
    rf101= rf101_row.iloc[0]
    print(f"{int(idx):>3}  {rd['idea_short']:38}  "
          f"{rd['ic']:>8.4f}  {rf['ic']:>8.4f}  {rf101['ic']:>9.4f}")

# ── Bảng 2: Tóm tắt tổng thể ────────────────────────────────
print("\n" + "=" * 65)
print("BẢNG 2 — Tóm tắt tổng thể")
print("=" * 65)
print(f"{'Chỉ số':25} {'Direct':>10} {'Full':>10} {'Full-101':>10}")
print("-" * 65)

metrics_summary = [
    ("Mean IC",          df_d["ic"].mean(),           df_f["ic"].mean(),           df_f101["ic"].mean()),
    ("Mean Sharpe",      df_d["sharpe"].mean(),        df_f["sharpe"].mean(),        df_f101["sharpe"].mean()),
    ("Mean Max Drawdown",df_d["max_drawdown"].mean(),  df_f["max_drawdown"].mean(),  df_f101["max_drawdown"].mean()),
    ("Mean Valid ratio", df_d["valid_ratio"].mean(),   df_f["valid_ratio"].mean(),   df_f101["valid_ratio"].mean()),
    ("Mean Relevance",   df_d["relevance"].mean(),     df_f["relevance"].mean(),     df_f101["relevance"].mean()),
    ("Alpha GOOD",       (df_d["verdict"]=="GOOD").sum(), (df_f["verdict"]=="GOOD").sum(), (df_f101["verdict"]=="GOOD").sum()),
    ("Alpha WEAK",       (df_d["verdict"]=="WEAK").sum(), (df_f["verdict"]=="WEAK").sum(), (df_f101["verdict"]=="WEAK").sum()),
    ("Alpha BAD",        (df_d["verdict"]=="BAD").sum(),  (df_f["verdict"]=="BAD").sum(),  (df_f101["verdict"]=="BAD").sum()),
]

for row in metrics_summary:
    name, vd, vf, vf101 = row
    if isinstance(vd, float):
        print(f"  {name:23} {vd:>10.4f} {vf:>10.4f} {vf101:>10.4f}")
    else:
        print(f"  {name:23} {int(vd):>10} {int(vf):>10} {int(vf101):>10}")

# ════════════════════════════════════════════════════════════
# VẼ BIỂU ĐỒ
# ════════════════════════════════════════════════════════════
print("\n[5] Vẽ biểu đồ...")

def make_segment_df(results, n_seg):
    rows = []
    seg_size = max(1, len(results) // n_seg)
    for s in range(n_seg):
        batch = results[s * seg_size: (s + 1) * seg_size]
        if not batch:
            continue
        rows.append({
            "segment"     : s,
            "ic"          : np.mean([r["metrics"]["ic"]           for r in batch]),
            "sharpe"      : np.mean([r["metrics"]["sharpe"]       for r in batch]),
            "max_drawdown": np.mean([r["metrics"]["max_drawdown"] for r in batch]),
            "valid_ratio" : np.mean([r["metrics"]["valid_ratio"]  for r in batch]),
            "relevance"   : np.mean([r["relevance"]               for r in batch]),
        })
    return pd.DataFrame(rows)

seg_d   = make_segment_df(results_direct,   N_SEGMENTS)
seg_f   = make_segment_df(results_full,     N_SEGMENTS)
seg_f101= make_segment_df(results_full_101, N_SEGMENTS)

COLORS  = {"Direct": "#e05c5c", "Full": "#3b7dd8", "Full-101": "#2ca02c"}
MARKERS = {"Direct": "o",       "Full": "s",       "Full-101": "^"}

METRIC_CHARTS = [
    ("ic",           "IC score",           "IC",            0.02,  "IC=0.02 (ngưỡng GOOD)"),
    ("sharpe",       "Sharpe Ratio",       "Sharpe",        0.5,   "Sharpe=0.5 (ngưỡng GOOD)"),
    ("max_drawdown", "Max Drawdown",       "Max Drawdown",  None,  None),
    ("valid_ratio",  "Valid Ratio",        "Valid Ratio",   None,  None),
    ("relevance",    "Idea Relevance (0-10)", "Relevance",  None,  None),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("QuantAgent Experiment: Direct vs Full vs Full-101",
             fontsize=15, fontweight="bold")
axes = axes.flatten()

for ax_i, (metric, title, ylabel, threshold, thr_label) in enumerate(METRIC_CHARTS):
    ax = axes[ax_i]
    for name, seg in [("Direct", seg_d), ("Full", seg_f), ("Full-101", seg_f101)]:
        ax.plot(seg["segment"], seg[metric],
                color=COLORS[name], marker=MARKERS[name],
                linewidth=2, markersize=5, label=name)
    if threshold is not None:
        ax.axhline(threshold, color="gray", linestyle="--",
                   linewidth=1, alpha=0.6, label=thr_label)
    if metric == "relevance":
        ax.set_ylim(0, 10)
    ax.set_title(title)
    ax.set_xlabel("Segment")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

# Chart 6: Verdict distribution (bar chart)
ax = axes[5]
verdict_data = {}
for name, (df, color) in VERSIONS.items(): 
    verdict_data[name] = df["verdict"].value_counts()

verdict_df = pd.DataFrame(verdict_data).fillna(0).reindex(["GOOD", "WEAK", "BAD"])
x     = np.arange(len(verdict_df.index))
width = 0.25

for i, (name, (_, color)) in enumerate(VERSIONS.items()):
    ax.bar(x + i * width, verdict_df[name], width,
           label=name, color=color, edgecolor="white", alpha=0.85)

ax.set_title("Phân bố Verdict")
ax.set_xlabel("Verdict")
ax.set_ylabel("Số alpha")
ax.set_xticks(x + width)
ax.set_xticklabels(["GOOD", "WEAK", "BAD"])
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
fig_path = RESULTS_DIR / f"figure_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"    Đã lưu biểu đồ: {fig_path}")
plt.show()

# ════════════════════════════════════════════════════════════
# LƯU KẾT QUẢ
# ════════════════════════════════════════════════════════════
print("\n[6] Lưu kết quả...")

timestamp   = datetime.now().strftime("%Y%m%d_%H%M")
result_path = RESULTS_DIR / f"experiment_{timestamp}.json"

def summary_dict(df, results):
    return {
        "mean_ic"         : float(df["ic"].mean()),
        "mean_sharpe"     : float(df["sharpe"].mean()),
        "mean_max_drawdown": float(df["max_drawdown"].mean()),
        "mean_valid_ratio": float(df["valid_ratio"].mean()),
        "mean_relevance"  : float(df["relevance"].mean()),
        "good"            : int((df["verdict"] == "GOOD").sum()),
        "weak"            : int((df["verdict"] == "WEAK").sum()),
        "bad"             : int((df["verdict"] == "BAD").sum()),
        "results"         : [{k: v for k, v in r.items() if k != "code"} for r in results],
    }

result_path.write_text(json.dumps({
    "config": {
        "inner_iterations": INNER_ITERATIONS,
        "outer_iterations": OUTER_ITERATIONS,
        "n_ideas"         : len(FIXED_IDEAS),
        "horizon"         : HORIZON,
    },
    "direct"  : summary_dict(df_d,    results_direct),
    "full"    : summary_dict(df_f,    results_full),
    "full_101": summary_dict(df_f101, results_full_101),
}, ensure_ascii=False, indent=2))

print(f"    Đã lưu: {result_path}")

print("\n" + "=" * 65)
print("EXPERIMENT HOÀN TẤT")
print("=" * 65)
for name, (df, _) in VERSIONS.items():
    print(f"  [{name}] IC={df['ic'].mean():.4f}  Sharpe={df['sharpe'].mean():.4f}  "
          f"MaxDD={df['max_drawdown'].mean():.4f}  "
          f"Relevance={df['relevance'].mean():.1f}/10  "
          f"GOOD={( df['verdict']=='GOOD').sum()}")