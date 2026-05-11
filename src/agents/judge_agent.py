from pathlib import Path
import re
import yaml
from loguru import logger
from .base_agent import BaseAgent
from ..kb.knowledge_base import KnowledgeBase


_PROMPT_PATH = Path(__file__).parent / "prompts" / "judge_prompt.yaml"


def _load_prompt() -> dict:
    if not _PROMPT_PATH.exists():
        return {
            "system": "Ban la chuyen gia review alpha factor. Tra loi dung dinh dang SCORE va FEEDBACK.",
            "user_template": "Y tuong giao dich goc: {idea}\n\nCode alpha can danh gia:\n```python\n{code}\n```",
            "model_config": {"temperature": 0.2, "max_tokens": 314},
        }
    with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_PROMPT_CFG = _load_prompt()


class JudgeAgent(BaseAgent):
    def __init__(self, kb: KnowledgeBase, model=None, api_key = None):
        super().__init__(model=model, agent_name="judge_agent", api_key=api_key)
        self.kb = kb
        self.prompt_cfg = _PROMPT_CFG

    def _build_kb_context(self, idea: str) -> str:
        examples = self.kb.retrieve_similar(idea, top_k=2)
        if not examples:
            return ""

        blocks = []
        for idx, rec in enumerate(examples, start=1):
            blocks.append(
                f"[KB {idx}] {rec.name}\n"
                f"Idea: {rec.idea[:200]}\n"
                f"IC={rec.metrics.ic:.4f}, Sharpe={rec.metrics.sharpe:.4f}\n"
                f"Code:\n```python\n{rec.code[:500]}\n```"
            )
        template = "\n\n".join(blocks)
        return f"""====== ALPHA THAM KHẢO ======
                So sánh code được đánh giá với các alpha dưới đây để đối chiếu logic và pattern:
                    {template}"""

    def judge(self, idea: str, code: str) -> tuple[float, str]:
        system_prompt = self.prompt_cfg.get("system", "")
        user_template = self.prompt_cfg.get("user_template", "")
        model_cfg = self.prompt_cfg.get("model_config", {})

        kb_context = self._build_kb_context(idea)

        prompt = user_template.format(idea=idea,kb_context=kb_context, code=code)
 
        resp = self._chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ], temperature=model_cfg.get("temperature", 0.2), max_tokens=model_cfg.get("max_tokens", 1024), max_retries=3)

        score_10 = 5.0
        m = re.search(r"SCORE:\s*([0-9]+(?:\.[0-9]+)?)", resp)
        if m:
            try:
                score_10 = max(0.0, min(10.0, float(m.group(1))))
            except ValueError:
                pass

        feedback = "Khong phan tich duoc"
        m = re.search(r"FEEDBACK:\s*([\s\S]*?)(?:\n\s*VẤN ĐỀ CHÍNH CẦN SỬA:|\n\s*VAN DE CHINH CAN SUA:|\Z)", resp)
        if m:
            feedback = " ".join(m.group(1).strip().split())
        else:
            lines = [line.strip() for line in resp.splitlines() if line.strip() and not line.strip().startswith("SCORE:")]
            if lines:
                feedback = " ".join(lines[:2])

        score = round(score_10 / 10.0, 4)
        logger.debug(f"Judge score={score:.2f}: {feedback[:200]}")
        return score, feedback
