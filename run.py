"""
QuantAgent - Entry point duy nhất.
Chạy: python run.py
Truy cập dashboard: http://localhost:8000
API docs: http://localhost:8000/api/docs
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import yaml
import uvicorn
import multiprocessing

if __name__ == "__main__":
    cfg_file = Path("config/settings.yaml")
    host, port, reload = "127.0.0.1", 8900, True
    if cfg_file.exists():
        cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
        host   = cfg.get("api", {}).get("host", host)
        port   = cfg.get("api", {}).get("port", port)
        reload = cfg.get("api", {}).get("reload", reload)

    print(f"""
╔══════════════════════════════════════════════╗
║           QuantAgent Vietnam v1.0            ║
╠══════════════════════════════════════════════╣
║  Dashboard : http://{host}:{port}            ║
║  API Docs  : http://{host}:{port}/api/docs   ║
╚══════════════════════════════════════════════╝
    """)

    try:
        uvicorn.run(
            "src.api.api_server:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )
    finally:
        # Cleanup multiprocessing resources to avoid leaked semaphore warning
        try:
            multiprocessing.current_process()._cleanup()
        except AttributeError:
            pass  # _cleanup may not be available in all Python versions