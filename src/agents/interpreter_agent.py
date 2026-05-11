from dataclasses import dataclass
from typing import Dict, Any
import math

from .base_agent import BaseAgent


def safe_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    if isinstance(x, float) and math.isnan(x):
        return default
    try:
        return float(x)
    except Exception:
        return default


def clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


@dataclass
class Classification:
    ic_label: str
    ic_pct: str
    sharpe_label: str
    wr_pct: str
    dd_label: str
    verdict: str
    confidence: str
    action_1: str
    action_2: str


def classify(ic: float, sharpe: float, wr: float, dd: float) -> Classification:
    ic_pct_val = clamp(50 + ic * 100, 50, 60)
    ic_pct = f"{ic_pct_val:.2f}%"

    if ic > 0.03:
        ic_label = "TÍCH CỰC"
    elif ic > 0.01:
        ic_label = "TRUNG TÍNH"
    else:
        ic_label = "YẾU"

    sharpe_label = (
        "CAO" if sharpe > 2 else
        "TRUNG BÌNH" if sharpe > 1 else
        "THẤP"
    )

    wr_pct = f"{wr * 100:.2f}%"
    dd_abs = abs(dd)
    dd_label = (
        "CAO" if dd_abs > 0.3 else
        "TRUNG BÌNH" if dd_abs > 0.1 else
        "THẤP"
    )

    score = ic * 0.4 + sharpe * 0.4 + wr * 0.2
    if score > 1.5:
        verdict, confidence = "KHẢ QUAN", "CAO"
        action_1 = "Giải ngân thử nghiệm 5–10% NAV"
        action_2 = "Thiết lập stop-loss 3%"
    elif score > 1.0:
        verdict, confidence = "TRUNG LẬP", "TRUNG BÌNH"
        action_1 = "Giải ngân nhỏ 2–3% NAV"
        action_2 = "Không sử dụng margin"
    else:
        verdict, confidence = "THẬN TRỌNG", "THẤP"
        action_1 = "Tiếp tục theo dõi tín hiệu"
        action_2 = "Chưa khuyến nghị giao dịch"

    return Classification(
        ic_label, ic_pct,
        sharpe_label, wr_pct,
        dd_label,
        verdict, confidence,
        action_1, action_2,
    )


SYSTEM_PROMPT = """Bạn là chuyên gia phân tích định lượng. Viết ngắn gọn, chuyên nghiệp, giải thích rõ alpha có bám ý tưởng gốc hay không, công thức đang làm gì và metrics nói lên điều gì."""


class AlphaInterpreterAgent(BaseAgent):
    def __init__(self, model: str | None = None):
        super().__init__(
            model=model or "groq/llama-3.3-70b-versatile",
            agent_name="interpreter_agent",
        )

    def _extract_formula_lines(self, latex: str) -> tuple[str, list[str]]:
        if not latex:
            return "", []
        lines = [line.strip() for line in latex.split("\n") if line.strip()]
        formula_only = "\n".join(line for line in lines if not line.startswith("%"))
        legends = [line.lstrip("%").strip() for line in lines if line.startswith("%")]
        return formula_only, legends

    def _build_alignment_hint(self, idea: str, code: str) -> str:
        idea_l = idea.lower()
        code_l = code.lower()
        
        # Bản đồ tri thức tài chính để đối chiếu logic
        knowledge_base = [
            {
                "category": "Động lượng giá (Price Momentum)",
                "keywords": ["momentum", "tăng dần", "suy yếu", "liên tiếp", "tốc độ", "xu hướng", "mạnh hơn", "tăng tốc"],
                "components": [
                    {"func": "pct_change", "reason": "đo lường tỷ suất sinh lời để xác định tốc độ biến động giá"},
                    {"func": "diff", "reason": "tính toán mức thay đổi tuyệt đối giữa các phiên giao dịch"},
                    {"func": "rolling", "reason": "thiết lập cửa sổ thời gian để quan sát quán tính của giá"}
                ],
                "insight": "Chiến lược Momentum kỳ vọng rằng xu hướng hiện tại sẽ tiếp tục duy trì trong tương lai gần."
            },
            {
                "category": "Đảo chiều về trung bình (Mean Reversion)",
                "keywords": ["đảo chiều", "xa trung bình", "quá mua", "quá bán", "bollinger", "rsi", "z-score", "lệch xa"],
                "components": [
                    {"func": "zscore", "reason": "chuẩn hóa độ lệch thống kê để tìm các vùng giá cực đoan"},
                    {"func": "std", "reason": "đo lường độ biến động để xác định các dải biên rủi ro"},
                    {"func": "mean", "reason": "xác định giá trị cân bằng mà cổ phiếu có xu hướng quay lại"}
                ],
                "insight": "Mean Reversion dựa trên giả thuyết rằng giá cả sẽ điều chỉnh khi đi quá xa giá trị nội tại."
            },
            {
                "category": "Phân tích khối lượng (Volume Analysis)",
                "keywords": ["volume", "thanh khoản", "khối lượng", "spike", "dòng tiền", "phân phối", "đột biến", "hấp thụ"],
                "components": [
                    {"func": "volume", "reason": "sử dụng thanh khoản làm biến xác nhận cho hành động giá"},
                    {"func": "rolling_mean", "reason": "so sánh khối lượng hiện tại với mức bình quân lịch sử"},
                    {"func": "v_ratio", "reason": "dùng tỷ lệ khối lượng để nhận diện sự tham gia của dòng tiền lớn"}
                ],
                "insight": "Khối lượng giao dịch phản ánh sự đồng thuận; giá tăng kèm khối lượng thấp là tín hiệu yếu."
            },
            {
                "category": "Cấu trúc nến (Candlestick Analysis)",
                "keywords": ["nến", "thân dài", "doji", "rút chân", "hammer", "gap", "mở cửa", "đóng cửa"],
                "components": [
                    {"func": "open", "reason": "sử dụng giá mở cửa để tính toán biên độ và khoảng trống giá"},
                    {"func": "high", "reason": "xác định mức giá cao nhất để đánh giá áp lực bán phía trên"},
                    {"func": "low", "reason": "xác định mức giá thấp nhất để đánh giá lực cầu bắt đáy"}
                ],
                "insight": "Mối quan hệ giữa các mức giá trong phiên phản ánh tâm lý chi phối của phe mua hoặc phe bán."
            }
        ]

        hits = []
        misses = []
        found_category = False

        for entry in knowledge_base:
            if any(k in idea_l for k in entry["keywords"]):
                found_category = True
                category_hits = []
                
                for comp in entry["components"]:
                    if comp["func"] in code_l:
                        category_hits.append(f"hàm {comp['func']} ({comp['reason']})")
                
                if category_hits:
                    hits.append(
                        f"{entry['category']}: Công thức đã thực thi qua {', '.join(category_hits)}. "
                        f"Nguyên lý: {entry['insight']}"
                    )
                else:
                    misses.append(
                        f"{entry['category']}: Ý tưởng đề cập đến '{entry['keywords'][0]}' "
                        f"nhưng mã nguồn chưa sử dụng các hàm phù hợp như {entry['components'][0]['func']}."
                    )

        if not found_category:
            return "Cảnh báo: Không thể xác định mối liên hệ logic. Ý tưởng cần bổ sung các từ khóa kỹ thuật cụ thể."
        report = ""
        if hits:
            report += "Các điểm tương thích (Hits):\n" + "\n".join([f"- {h}" for h in hits]) + "\n\n"
        if misses:
            report += "Các điểm thiếu hụt (Misses):\n" + "\n".join([f"- {m}" for m in misses]) + "\n\n"
        
        if hits and not misses:
            report += "Kết luận: Mã nguồn phản ánh hoàn toàn chính xác ý tưởng đầu tư ban đầu."
        elif hits and misses:
            report += "Kết luận: Mã nguồn có bám sát ý tưởng nhưng cần bổ sung thêm các biến số để tối ưu hóa logic."
        else:
            report += "Kết luận: Có sự sai lệch đáng kể giữa mô tả ý tưởng và công thức triển khai thực tế."

        return report

    def _build_backtest_reading(self, metrics: Dict[str, Any], cls: Classification) -> str:
        ic = safe_float(metrics.get("ic"))
        sharpe = safe_float(metrics.get("sharpe"))
        wr = safe_float(metrics.get("win_rate"))
        ls = safe_float(metrics.get("long_short_spread"))
        top_bucket = safe_float(metrics.get("top_bucket_return"))
        bottom_bucket = safe_float(metrics.get("bottom_bucket_return"))
        liq = safe_float(metrics.get("avg_liquidity"))

        parts = [
            f"IC {ic:.4f} cho thấy mức độ dự báo {cls.ic_label.lower()}.",
            f"Sharpe {sharpe:.2f} phản ánh độ ổn định ở mức {cls.sharpe_label.lower()}.",
            f"Win rate {wr * 100:.1f}% cho biết xác suất tín hiệu đúng chưa thật sự áp đảo." if wr < 0.55 else f"Win rate {wr * 100:.1f}% cho thấy tín hiệu thắng khá ổn.",
            f"Top bucket return {top_bucket:.4f}, bottom bucket return {bottom_bucket:.4f}, long-short spread {ls:.4f} {'ủng hộ khả năng ranking' if ls > 0 else 'chưa cho thấy lợi thế ranking rõ'}.",
            f"Thanh khoản trung bình {liq:.2f} {'hỗ trợ triển khai' if liq > 1 else 'cần thận trọng khi triển khai thực chiến'}."
        ]
        return " ".join(parts)

    def interpret(self,
                  idea: str,
                  symbol: str,
                  metrics: Dict[str, Any],
                  code: str,
                  latex: str = "") -> Dict[str, Any]:
        if not metrics or safe_float(metrics.get("n_days")) < 5:
            return {
                "report_text": "⚠️ Không đủ dữ liệu backtest.",
                "summary": "Không đủ dữ liệu backtest.",
                "idea_alignment": "Chưa đủ dữ liệu để đánh giá.",
                "formula_explanation": "Không có dữ liệu công thức.",
                "backtest_reading": "Không đủ dữ liệu backtest.",
                "risks": ["Số ngày hiệu lực quá ít nên kết luận không đáng tin."],
                "actions": ["Tăng dữ liệu backtest trước khi đưa ra quyết định."],
                "improvements": ["Sinh lại alpha với cấu trúc tín hiệu rõ hơn."],
            }

        ic = safe_float(metrics.get("ic"))
        sharpe = safe_float(metrics.get("sharpe"))
        wr = safe_float(metrics.get("win_rate"))
        dd = safe_float(metrics.get("max_drawdown"))
        ls = safe_float(metrics.get("long_short_spread"))
        top_bucket = safe_float(metrics.get("top_bucket_return"))
        bottom_bucket = safe_float(metrics.get("bottom_bucket_return"))
        liq = safe_float(metrics.get("avg_liquidity"))
        n_days = int(safe_float(metrics.get("n_days"), 0))

        cls = classify(ic, sharpe, wr, dd)
        formula_only, legends = self._extract_formula_lines(latex)
        alignment_hint = self._build_alignment_hint(idea, code)
        backtest_reading = self._build_backtest_reading(metrics, cls)

        prompt = f"""
Viết ngắn gọn 5 phần sau bằng tiếng Việt, súc tích và chuyên nghiệp:
1. Tóm tắt 2 câu về alpha này.
2. Độ bám idea: alpha đang bám hypothesis gốc ở đâu, lệch ở đâu.
3. Giải thích công thức: công thức đang biến ý tưởng thành tín hiệu thế nào.
4. Rủi ro chính: 2 ý.
5. Hướng cải thiện: 2 ý cụ thể.

Ý tưởng: {idea[:300]}
Công thức: {formula_only or '(không có công thức)'}
Giải thích ký hiệu: {'; '.join(legends[:6]) if legends else '(không có)'}
Metrics: IC={ic:.4f}, Sharpe={sharpe:.2f}, WinRate={wr:.2%}, MaxDD={dd:.4f}, LongShort={ls:.4f}, TopBucket={top_bucket:.4f}, BottomBucket={bottom_bucket:.4f}, AvgLiquidity={liq:.2f}, NDays={n_days}
Gợi ý độ bám idea: {alignment_hint}
Gợi ý đọc backtest: {backtest_reading}
"""

        llm_text = self._chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=700,
            temperature=0.3,
        )

        summary = f"Alpha cho ý tưởng '{idea[:100]}' hiện ở trạng thái {cls.verdict.lower()} với độ tin cậy {cls.confidence.lower()}. IC {ic:.4f}, Sharpe {sharpe:.2f}, long-short spread {ls:.4f}."
        formula_explanation = formula_only or "Không trích xuất được công thức LaTeX."
        if legends:
            formula_explanation += "\n" + "\n".join(f"- {x}" for x in legends[:6])

        risks = []
        if ic < 0.01:
            risks.append("IC thấp, tín hiệu dự báo chưa đủ mạnh.")
        if ls <= 0:
            risks.append("Long-short spread không dương, alpha chưa phân hạng cổ phiếu tốt.")
        if liq < 1.0:
            risks.append("Thanh khoản trung bình thấp, dễ bị trượt giá khi triển khai.")
        if dd < -0.1:
            risks.append("Drawdown tương đối cao so với chất lượng tín hiệu hiện tại.")
        if not risks:
            risks.append("Rủi ro chính nằm ở việc alpha có thể suy yếu khi regime thị trường thay đổi.")

        actions = [cls.action_1, cls.action_2]
        if ls > 0 and ic > 0.01:
            actions.append("Có thể dùng alpha này như tín hiệu ranking trong danh mục thử nghiệm.")
        else:
            actions.append("Chưa nên dùng độc lập; phù hợp hơn để tiếp tục refine.")

        improvements = []
        if "volume" not in code.lower() and "khối lượng" in idea.lower():
            improvements.append("Bổ sung thành phần volume để bám idea hơn và giảm drift khỏi hypothesis gốc.")
        if ".rank(axis=1, pct=True) - 0.5" not in code:
            improvements.append("Chuẩn hóa cross-sectional rõ hơn ở bước cuối để tăng khả năng ranking.")
        if ls <= 0:
            improvements.append("Ưu tiên chỉnh tham số hoặc đổi cấu trúc tín hiệu để top bucket vượt rõ bottom bucket.")
        if not improvements:
            improvements.append("Thử tinh chỉnh window và thêm filter thanh khoản để cải thiện độ ổn định.")
            improvements.append("Đánh giá thêm theo từng regime thị trường để xác định vùng alpha hoạt động tốt nhất.")

        report_text = "\n\n".join([
            summary,
            f"Độ bám idea: {alignment_hint}",
            f"Đọc backtest: {backtest_reading}",
            f"Khuyến nghị: {' | '.join(actions)}",
            llm_text.strip(),
        ])

        return {
            "report_text": report_text,
            "summary": summary,
            "idea_alignment": alignment_hint,
            "formula_explanation": formula_explanation,
            "backtest_reading": backtest_reading,
            "risks": risks[:3],
            "actions": actions[:3],
            "improvements": improvements[:3],
            "raw_llm_text": llm_text.strip(),
        }
