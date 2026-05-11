import re


def is_valid_python(code: str) -> tuple[bool, str]:
    try:
        compile(code, "<check>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, str(e)


def extract_code_block(text: str) -> str:
    # Lấy tất cả block ```python ... ```
    blocks = re.findall(r'```python\s*([\s\S]*?)\s*```', text)
    if blocks:
        best = max(blocks, key=len)
        if len(best) > 50:
            return best.strip()

    # Fallback: block ``` không có tag python
    blocks = re.findall(r'```\s*([\s\S]*?)\s*```', text)
    if blocks:
        best = max(blocks, key=len)
        if len(best) > 50:
            return best.strip()

    # Fallback cuối: text thuần có dấu hiệu là code
    stripped = text.strip()
    if any(kw in stripped for kw in ('class ', 'def ', 'import ')):
        return stripped

    return ""