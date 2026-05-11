# REPO_SUMMARY

Tổng hợp trích đoạn quan trọng để LLM hiểu nhanh kiến trúc và luồng chính. Các liên kết tham chiếu theo đường dẫn repo.

## 1) API entry & routing
- [src/api/api_server.py](src/api/api_server.py): FastAPI entry, mount CORS, include routers `/api/alpha`, `/api/kb`, `/api/backtest`, `/api/predict`; serve dashboard tĩnh.
- [src/api/routes/alpha_routes.py](src/api/routes/alpha_routes.py): Endpoint `/mine` chạy Inner + Outer Loop, build OuterLoop với WriterAgent, JudgeAgent, ReviewerAgent, BacktestEngine, CodeRunner, KnowledgeBase. Trả về code, LaTeX (render thêm bằng `code_to_latex`), metrics, outer_history. Endpoint `/symbols` liệt kê CSV trong `data/raw`.
- [src/api/routes/backtest_routes.py](src/api/routes/backtest_routes.py): `/run` backtest thủ công. Nạp dữ liệu, preprocess, BacktestEngine + CodeRunner, Evaluator để tính metrics.
- [src/api/routes/predict_routes.py](src/api/routes/predict_routes.py): `/run` dự báo; dùng EnsemblePredictor (kb + engine), tùy chọn retrain.
- [src/api/routes/kb_routes.py](src/api/routes/kb_routes.py): liệt kê KB (lọc min_ic, top_k), thống kê avg/best ic/sharpe.

## 2) Vòng lặp sinh alpha
- [src/agents/inner_loop.py](src/agents/inner_loop.py): InnerLoop gọi WriterAgent sinh (code, LaTeX), validate syntax, JudgeAgent chấm điểm, giữ best code/latex/score, dừng khi score >= threshold.
- [src/backtest/outer_loop.py](src/backtest/outer_loop.py): OuterLoop lặp qua inner, load alpha class (CodeRunner), compute_signals & evaluate (BacktestEngine), ReviewerAgent phản hồi; chọn best IC, nếu IC >= min_ic_save thì lưu vào KnowledgeBase (AlphaRecord/AlphaMetrics). Lưu lịch sử iter.
- [src/agents/writer_agent.py](src/agents/writer_agent.py): Prompt thư viện alpha mẫu + quy trình sinh LaTeX trước, code sau; lấy ví dụ tương tự từ KB; trả (code, latex). Model mặc định `groq/llama-3.1-8b-instant`.
- [src/agents/judge_agent.py](src/agents/judge_agent.py): Chấm điểm 0-1 dựa trên logic, look-ahead, NaN, liên tục; parse SCORE/COMMENT từ phản hồi LLM.
- [src/agents/reviewer_agent.py](src/agents/reviewer_agent.py): Nhận metrics + code, nhờ LLM gợi ý cải tiến ngắn gọn tiếng Việt; fallback nếu IC=0.
- [src/agents/base_agent.py](src/agents/base_agent.py): Wrapper litellm, retry 429 với backoff; model mặc định `groq/llama-3.1-8b-instant`.

## 3) Backtest & dữ liệu
- [src/backtest/backtest_engine.py](src/backtest/backtest_engine.py): `compute_signals` chuẩn hóa output DataFrame; `evaluate` tính IC daily (Spearman), Sharpe(IC), win_rate, max_drawdown trên cumulative IC, valid_ratio; zero_metrics khi thiếu dữ liệu.
- [src/utils/code_runner.py](src/utils/code_runner.py): Exec code alpha trong namespace an toàn (pd/np/AlphaBase); tìm class có `calc`; `safe_run` để chạy đoạn mã khác.
- [src/data/data_loader.py](src/data/data_loader.py): Đọc tất cả `data/raw/stock_data_*.csv` (các symbol), ghép thành dict DataFrame open/high/low/close/volume.
- [src/data/preprocessor.py](src/data/preprocessor.py): Cắt khoảng thời gian, ffill + fillna, thêm `returns`.
- [src/utils/validator.py](src/utils/validator.py): Kiểm tra syntax Python, trích code block từ markdown.

## 4) Knowledge Base
- [src/kb/knowledge_base.py](src/kb/knowledge_base.py): Lưu/đọc kb.json, add AlphaRecord, list_all(min_ic), retrieve_similar lấy top theo IC.
- [src/kb/alpha_schema.py](src/kb/alpha_schema.py): Định nghĩa AlphaRecord (idea/code/metrics/score/review/time) và AlphaMetrics (ic, sharpe, max_drawdown, win_rate, valid_ratio).

## 5) Utils & cấu hình
- [src/utils/paths.py](src/utils/paths.py): Đường dẫn chuẩn: ROOT/data/raw|processed, kb_store, models, logs.
- [src/utils/logger.py](src/utils/logger.py): Cấu hình loguru ra stdout + logs/app.log.
- [src/utils/prompt_loader.py](src/utils/prompt_loader.py): Load prompt YAML cho agents (prompts nằm ở src/agents/prompts/*.yaml).
- [config/settings.yaml](config/settings.yaml), [config/backtest_config.yaml](config/backtest_config.yaml): cấu hình (chưa đọc chi tiết ở đây).

## 6) API schemas
- [src/api/schemas/request_schemas.py](src/api/schemas/request_schemas.py): Kiểu dữ liệu request (AlphaMineRequest, BacktestRequest, PredictRequest...).
- [src/api/schemas/response_schemas.py](src/api/schemas/response_schemas.py): Response models (AlphaMineResponse, MetricsOut, KBListResponse...).

## 7) ML dự báo
- [src/ml/ensemble_predictor.py](src/ml/ensemble_predictor.py): Ensemble từ KB + engine cho dự báo (chưa trích chi tiết).
- [src/ml/model_registry.py](src/ml/model_registry.py), [src/ml/feature_builder.py](src/ml/feature_builder.py): khung model/feature (chưa trích).

## 8) Khác
- Dữ liệu mẫu: nhiều CSV trong `data/raw/stock_data_*.csv`.
- Prompts tùy biến: `src/agents/prompts/*.yaml` (writer/judge/reviewer).

Ghi chú: Các module chưa mở chi tiết (ml/ensemble_predictor.py, model_registry.py, feature_builder.py, config/*.yaml, prompts/*.yaml) cần đọc thêm nếu muốn LLM nắm trọn vẹn cấu hình/feature. Nội dung trên bao phủ luồng chính sinh alpha, backtest và API. 
