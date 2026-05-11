import litellm
import time
from loguru import logger

FORMULA_PROMPT = '''Chuyển đoạn code Python alpha factor sang công thức toán học LaTeX.
Chỉ trả về LaTeX, không giải thích, không code.

Ví dụ:
Code: -close.pct_change(5).rolling(3).mean()
LaTeX: \\alpha_t = -\\frac{1}{3}\\sum_{k=0}^{2}\\frac{P_{t-k} - P_{t-k-5}}{P_{t-k-5}}

Code: (close - close.rolling(20).mean()) / close.rolling(20).std()
LaTeX: \\alpha_t = \\frac{P_t - \\mu_{20}(P)}{\\sigma_{20}(P)}

Code: volume / volume.rolling(10).mean() - 1
LaTeX: \\alpha_t = \\frac{V_t}{\\bar{V}_{10}} - 1
'''

def code_to_latex(code: str, model: str = "groq/llama-3.1-8b-instant") -> str:
    """Dịch Python alpha code sang LaTeX formula."""
    # Chỉ lấy phần calc() để ngắn gọn
    calc_lines = []
    in_calc = False
    for line in code.splitlines():
        if "def calc" in line:
            in_calc = True
            continue
        if in_calc:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                calc_lines.append(stripped)
    calc_code = "\n".join(calc_lines[:8])  # tối đa 8 dòng

    try:
        resp = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": FORMULA_PROMPT},
                {"role": "user", "content": f"Code:\n{calc_code}\n\nLaTeX:"},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        latex = resp.choices[0].message.content.strip()
        # Làm sạch nếu LLM bọc thêm ``` hoặc $
        latex = latex.replace("```latex", "").replace("```", "").strip()
        latex = latex.strip("$").strip()
        logger.debug(f"LaTeX: {latex[:80]}")
        return latex
    except Exception as e:
        logger.warning(f"formula_renderer lỗi: {e}")
        return ""
