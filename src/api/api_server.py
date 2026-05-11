"""
FastAPI entry point. Serve cả REST API và HTML dashboard.
Chạy: python run.py → http://127.0.0.1:8000
Dashboard: http://127.0.0.1:8000/dashboard
"""
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .routes.alpha_routes import router as alpha_router
from .routes.kb_routes import router as kb_router
from .routes.backtest_routes import router as backtest_router
from .routes.predict_routes import router as predict_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi động
    print("🚀 QuantAgent API đang khởi động...")
    yield
    print("👋 QuantAgent API đã tắt")


app = FastAPI(
    title="QuantAgent API",
    description="API cho hệ thống khai thác alpha factor thị trường VN",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS cho phép frontend HTML gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký các router
app.include_router(alpha_router, prefix="/api/alpha", tags=["Alpha Mining"])
app.include_router(kb_router, prefix="/api/kb", tags=["Knowledge Base"])
app.include_router(backtest_router, prefix="/api/backtest", tags=["Backtest"])
app.include_router(predict_router, prefix="/api/predict", tags=["Prediction"])

# Serve static files (CSS, JS, assets)
TEMPLATES_DIR = Path("templates")
if TEMPLATES_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(TEMPLATES_DIR / "assets")), name="assets")


@app.get("/")
async def root():
    return {"status": "ok", "message": "QuantAgent API đang chạy. Truy cập /dashboard"}


@app.get("/dashboard")
async def dashboard():
    """Serve trang dashboard chính."""
    index_path = TEMPLATES_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    raise HTTPException(404, "Dashboard HTML chưa được tạo")


@app.get("/dashboard/{page}")
async def dashboard_page(page: str):
    """Serve các trang con của dashboard."""
    page_path = TEMPLATES_DIR / f"{page}.html"
    if page_path.exists():
        return FileResponse(str(page_path))
    raise HTTPException(404, f"Trang {page} không tồn tại")


@app.get("/api/health")
@app.get("/health")
async def health():
    return {"status": "healthy"}