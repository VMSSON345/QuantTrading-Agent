from pathlib import Path
import re
import yaml
from loguru import logger
from .base_agent import BaseAgent
from .context_buffer import ContextBuffer
from ..kb.knowledge_base import KnowledgeBase
from ..kb.alpha_101 import Alpha101KB
from ..utils.validator import extract_code_block
from dotenv import load_dotenv
import os

load_dotenv()


_PROMPT_PATH = Path(__file__).parent / "prompts" / "writer_prompt.yaml"


def _load_prompt() -> dict:
    if not _PROMPT_PATH.exists():
        return {
            "system": "Ban la chuyen gia quant viet alpha factor cho thi truong chung khoan Viet Nam.",
            "user_template": "Y tuong giao dich: {idea}\n{kb_context}\n{feedback_section}",
            "kb_context_template": "Alpha tuong tu trong KB:\n{similar_alphas}",
            "kb_context_empty": "(KB chua co alpha tuong tu)",
            "feedback_section_first": "(Lan sinh dau tien)",
            "feedback_section_refine": "Feedback: {judge_feedback}\n\nCode cu:\n```python\n{previous_code}\n```",
            "model_config": {"temperature": 0.5, "max_tokens": 1600},
        }
    with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_PROMPT_CFG = _load_prompt()


def extract_latex_block(text: str) -> str:
    m = re.search(r'```latex\s*([\s\S]*?)\s*```', text)
    if m:
        return m.group(1).strip()
    m = re.search(r'(\\alpha_t\s*=[\s\S]{5,300}?)(?:\n\n|```|\Z)', text)
    if m:
        return m.group(1).strip()
    return ""


def extract_latex_legend(latex: str) -> str:
    """Tách phần chú thích ký hiệu từ LaTeX block (dòng bắt đầu bằng %)."""
    lines = latex.split("\n")
    legend_lines = [l.lstrip("%").strip() for l in lines if l.strip().startswith("%")]
    return "\n".join(legend_lines) if legend_lines else ""


class WriterAgent(BaseAgent):
    def __init__(self, kb: KnowledgeBase, alpha101_kb: Alpha101KB, model=None, api_key = None):
        super().__init__(model=model or "groq/meta-llama/llama-4-scout-17b-16e-instruct",
                         agent_name="writer_agent", api_key=api_key)
        self.kb = kb
        self.prompt_cfg = _PROMPT_CFG
        self.alpha101_kb = alpha101_kb

    def _build_system_prompt(self) -> str:
        # RUNTIME đã baked vào yaml — không cần nối chuỗi nữa
        return self.prompt_cfg.get("system", "").strip()

    def _build_kb_context(self, idea: str) -> str:
        examples = self.kb.retrieve_similar(idea, top_k=1) 
        if not examples:
            return self.prompt_cfg.get("kb_context_empty", "")

        similar_alphas = []
        for idx, rec in enumerate(examples, start=1):
            similar_alphas.append(
                f"[{idx}] {rec.name}\n"
                f"Idea: {rec.idea[:200]}\n"
                f"IC={rec.metrics.ic:.4f}, Sharpe={rec.metrics.sharpe:.4f}\n"
                f"Code:\n```python\n{rec.code[:500]}\n```"
                f"Score: {rec.judge_score}\n"
                f"Comment: {rec.review_comment}"
            )

        template = "\n\n".join(similar_alphas)
        return f"""====== ALPHA THAM KHẢO (KB nội bộ) ======
            Các alpha dưới đây đã chạy thực tế và có IC tốt. Học pattern, tạo biến thể cải tiến: 
            {template}"""
    
    def _build_alpha101_context(self, idea: str) -> str:
        """Alpha101 KB — WorldQuant formulas, chỉ dùng làm tham khảo cấu trúc."""
        if self.alpha101_kb is None:
            return self.prompt_cfg.get("alpha101_context_empty", "")
    
        refs = self.alpha101_kb.retrieve(idea, top_k=1)
        if not refs:
            return self.prompt_cfg.get("alpha101_context_empty", "")
    
        rows = []
        for idx, r in enumerate(refs, start=1):
            rows.append(
                f"[{idx}] {r.id} — inputs: {', '.join(r.inputs)}\n"
                f"Code:\n```python\n{r.code[:350]}\n```"
            )
    
        template = "\n\n".join(rows)
        return f"""====== WORLDQUANT 101 ALPHAS TƯƠNG TỰ ======
                Các công thức dưới đây có cấu trúc gần với ý tưởng. Dùng làm tham khảo kỹ thuật:
                    {template}"""

    def _extract_previous_code(self, buffer: ContextBuffer) -> str:
        for msg in reversed(buffer.get()):
            if msg.get("role") != "assistant":
                continue
            code = extract_code_block(msg.get("content", ""))
            if code:
                return code
        return ""
    

    def _build_feedback_section(self, feedback: str, buffer: ContextBuffer) -> str:
        if not feedback:
            return self.prompt_cfg.get("feedback_section_first", "")

        previous_code = self._extract_previous_code(buffer) or "# Chua co code truoc do"
        template = self.prompt_cfg.get(
            "feedback_section_refine",
            "Feedback: {judge_feedback}\n\nCode cu:\n```python\n{previous_code}\n```",
        )
        return template.format(
            judge_feedback=feedback[:800],
            previous_code=previous_code[:1200],
        )

    def write(self, idea: str, buffer: ContextBuffer, feedback: str = "") -> tuple:
        """Returns: (code, latex)"""
        kb_context       = self._build_kb_context(idea)
        alpha101_context = self._build_alpha101_context(idea)
        feedback_section = self._build_feedback_section(feedback, buffer)
    
        user_template = self.prompt_cfg.get(
            "user_template",
            "{idea}\n{kb_context}\n{alpha101_context}\n{feedback_section}",
        )
        msg = user_template.format(
            idea=idea[:500],
            kb_context=kb_context,
            alpha101_context=alpha101_context,
            feedback_section=feedback_section,
        )
    
        msgs = (
            [{"role": "system", "content": self._build_system_prompt()}]
            + buffer.get()[-2:]
            + [{"role": "user", "content": msg}]
        )
    
        model_cfg = self.prompt_cfg.get("model_config", {})
        resp = self._chat(
            msgs,
            temperature=model_cfg.get("temperature", 0.75),
            max_tokens=model_cfg.get("max_tokens", 2048),
        )
    
        buffer.add("user", msg)
        buffer.add("assistant", resp)
    
        latex = extract_latex_block(resp)
        code  = extract_code_block(resp)
    
        if not latex:
            logger.warning("Writer: không extract được LaTeX")
        if not code:
            logger.warning("Writer: không extract được code")
    
        logger.debug(
            f"Writer: latex={'ok' if latex else 'miss'} | "
            f"code={len(code)}ch | "
            f"kb={'ok' if kb_context else 'empty'} | "
            f"alpha101={'ok' if alpha101_context else 'empty'}"
        )
        return code, latex
    def write_direct(self, idea: str) -> str:
        system_prompt = """Bạn là Quantitative Analyst chuyên thị trường chứng khoán Việt Nam.
        NHIỆM VỤ: Viết AlphaFactor từ ý tưởng giao dịch.

        QUY TẮC KỸ THUẬT BẮT BUỘC:
        1. KHÔNG look-ahead bias: Tuyệt đối không dùng shift(-n) với n > 0.

        2. Vectorized code: Sử dụng pandas/numpy, KHÔNG dùng vòng lặp for/while.

        3. Signal PHẢI là pd.DataFrame:
        - calc() BẮT BUỘC trả về pd.DataFrame với cùng index và columns như data["close"].
        - KHÔNG return ndarray, Series, list, hay scalar.
        - SAI:  return np.array(signal)
        - ĐÚNG: return signal.rank(axis=1, pct=True) - 0.5

        4. TUYỆT ĐỐI KHÔNG dùng .values hay .to_numpy():
        - Luôn làm việc trực tiếp trên DataFrame/Series để giữ nguyên index và columns.
        - SAI:  arr = close.values; result = arr / arr.mean()
        - ĐÚNG: result = close / close.mean()

        5. Khi cần rolling/window:
        - ĐÚNG: close.rolling(20).mean()       → DataFrame
        - SAI:  np.convolve(close.values, ...)  → ndarray

        6. Liên tục hóa signal — dòng cuối PHẢI là:
        return <signal>.rank(axis=1, pct=True) - 0.5
        - rank(pct=True) cho ra [0, 1], trừ 0.5 để center về [-0.5, 0.5].
        - KHÔNG thêm fillna(0) sau rank — giữ NaN nguyên.
        - KHÔNG return signal nhị phân 0/1.

        7. Xử lý NaN:
        - CHỈ dùng ffill() trên price/volume trước khi tính toán.
        - KHÔNG fillna(0) trên price/volume — làm sai mọi tính toán.
        - rank(axis=1, pct=True, na_option='keep') — giữ NaN, không fill sau rank.

        CẤU TRÚC CODE MẪU (tuân thủ chính xác):
        ```python
        import pandas as pd
        import numpy as np

        class AlphaBase:
            window_length = 20
            inputs = ["close"]
            def calc(self, data: dict) -> pd.DataFrame: pass

        class MyAlpha(AlphaBase):
            def __init__(self):
                self.window_length = 20
                self.inputs = ["close", "volume"]  # Tùy biến theo ý tưởng

            def calc(self, data: dict) -> pd.DataFrame:
                close  = data["close"].ffill()   # pd.DataFrame, KHÔNG .values
                volume = data["volume"].ffill()  # pd.DataFrame, KHÔNG .values

                # --- Logic tính toán (ví dụ) ---
                momentum = close / close.shift(self.window_length) - 1  # vẫn là DataFrame
                vol_filter = volume / volume.rolling(20).mean()          # vẫn là DataFrame
                signal = momentum * vol_filter                           # element-wise, vẫn là DataFrame

                # --- Bắt buộc: dòng cuối cùng ---
                return signal.rank(axis=1, pct=True) - 0.5  # pd.DataFrame, centered quanh 0
        ```
        """

        # 2. Tạo prompt người dùng đơn giản
       # 2. User Prompt: Tập trung vào Idea và định dạng đầu ra
        user_msg = f"""Ý tưởng giao dịch: {idea}

    YÊU CẦU ĐẦU RA: Viết code Python hoàn chỉnh theo cấu trúc AlphaBase trong block ```python ... ```

    Hãy thực hiện ngay:"""

        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ]

        # 3. Gọi model trực tiếp thông qua hàm _chat có sẵn của class
        model_cfg = self.prompt_cfg.get("model_config", {})
        resp = self._chat(
            msgs,
            temperature=model_cfg.get("temperature", 0.5), # Direct thường cần temp thấp cho ổn định
            max_tokens=model_cfg.get("max_tokens", 2000),
        )

        # 4. Extract code từ response
        code = extract_code_block(resp)
        
        if not code:
            logger.warning("Writer Direct: không extract được code từ response")
            return "" # Hoặc trả về resp tùy bạn
            
        return code
