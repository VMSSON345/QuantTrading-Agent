import yaml
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"

def load_config(name: str = "settings") -> dict:
    """Chi dung de load llm.yaml va settings.yaml — KHONG load prompt."""
    fp = CONFIG_DIR / f"{name}.yaml"
    if fp.exists():
        with open(fp, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}
