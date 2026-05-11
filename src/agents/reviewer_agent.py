from .base_agent import BaseAgent

REVIEWER_SYSTEM = """Ban la chuyen gia quant chuyên ve alpha factor cho thi truong chung khoan Viet Nam.

Nhiem vu: Doc metrics backtest va code alpha, dua ra de xuat cai thien CU THE va NGAN GON.

PHAN TICH THEO THU TU:
1. IC (Information Coefficient): do tuong quan signal vs return thuc
   - IC > 0.05: tot, giu nguyen huong, thu chinh tham so
   - IC 0.01-0.05: trung binh, thu them filter hoac ket hop signal
   - IC < 0.01: yeu, doi cach tiep can hoan toan
   - IC < 0: signal nguoc chieu, them dau tru vao cong thuc

2. Sharpe Ratio:
   - Sharpe < 0.5: signal qua noisy, thu rolling mean de lam mo hoac clip(-2,2)
   - Sharpe 0.5-1.0: on, thu giam lookback window
   - Sharpe > 1.0: tot

3. Win Rate:
   - WinRate < 50%: thu them dieu kien xac nhan (volume tang + momentum cung chieu)
   - WinRate > 55%: tot

4. Max Drawdown:
   - DD < -20%: rui ro cao, thu normalize hoac rank cross-sectional

QUY TAC DE XUAT:
- Moi de xuat phai LA CONG THUC CU THE, khong chung chung
- Vi du DUNG: "Them filter: nhan voi (volume / volume.rolling(10).mean()).clip(0,2)"
- Vi du SAI: "Can cai thien logic tinh toan"
- Toi da 3 de xuat, moi de xuat 1 dong
- Viet bang tieng Viet, ngan gon
"""


class ReviewerAgent(BaseAgent):
    def __init__(self, model=None, api_key = None):
        super().__init__(model=model, agent_name="reviewer_agent", api_key=api_key)

    def review(self, code: str, metrics: dict) -> str:
        if not metrics:
            return "Backtest khong chay duoc, kiem tra lai syntax va logic code."

        ic     = metrics.get("ic", 0)
        sharpe = metrics.get("sharpe", 0)
        dd     = metrics.get("max_drawdown", 0)
        wr     = metrics.get("win_rate", 0)
        n_days = metrics.get("n_days", 0)

        # Pre-check nhanh khong can goi LLM
        if ic == 0 and sharpe == 0:
            return "IC=0 va Sharpe=0: backtest that bai, kiem tra lai code co tinh duoc signal hop le khong."

        metrics_str = (
            f"IC={ic:.4f} | Sharpe={sharpe:.4f} | "
            f"MaxDD={dd:.4f} | WinRate={wr:.2%} | N_days={n_days}"
        )

        # Them nhan xet so bo de LLM co context
        quick_notes = []
        if ic < 0:
            quick_notes.append("IC am: signal dang nguoc chieu voi return thuc")
        if abs(ic) < 0.01:
            quick_notes.append("IC rat thap: can doi huong tiep can")
        if sharpe < 0.5:
            quick_notes.append("Sharpe thap: signal noisy")
        if wr < 0.50:
            quick_notes.append("WinRate duoi 50%: du bao sai nhieu hon dung")
        if dd < -0.20:
            quick_notes.append("Drawdown lon: can normalize hoac rank")

        notes_str = " | ".join(quick_notes) if quick_notes else "Ket qua kha on"

        resp = self._chat([
            {"role": "system", "content": REVIEWER_SYSTEM},
            {"role": "user", "content": (
                f"Metrics: {metrics_str}\n"
                f"Nhan xet so bo: {notes_str}\n\n"
                f"Code alpha (phan chinh):\n```python\n{code[:500]}\n```\n\n"
                f"De xuat cai thien cu the:"
            )},
        ])
        return resp
