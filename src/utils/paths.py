from pathlib import Path

ROOT_DIR      = Path(__file__).resolve().parents[2]
DATA_DIR      = ROOT_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
KB_DIR        = ROOT_DIR / "kb_store"
MODEL_DIR     = ROOT_DIR / "models"
LOG_DIR       = ROOT_DIR / "logs"
FRONTEND_DIR  = ROOT_DIR / "frontend"

# Tạo thư mục nếu chưa có
for d in [RAW_DIR, PROCESSED_DIR, KB_DIR, MODEL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)
