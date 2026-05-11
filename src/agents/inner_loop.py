from dataclasses import dataclass, field
from loguru import logger
from .context_buffer import ContextBuffer
from ..utils.validator import is_valid_python


@dataclass
class InnerResult:
    final_code:    str   = ""
    final_latex:   str   = ""
    final_score:   float = 0.0
    final_metrics: dict  = field(default_factory=dict)
    history:       list  = field(default_factory=list)


class InnerLoop:
    def __init__(self, writer, judge,  runner=None,
                 max_inner_iter=5, score_threshold=0.75):
        self.writer    = writer
        self.judge     = judge
        self.runner    = runner
        self.max_iter  = max_inner_iter
        self.threshold = score_threshold

    def run(self, idea: str, feedback: str = "") -> InnerResult:
        buffer = ContextBuffer()
        result = InnerResult()

        for i in range(self.max_iter):
            logger.info(f"  Inner iter {i+1}/{self.max_iter}")

            code, latex = self.writer.write(idea, buffer, feedback)

            valid, err = is_valid_python(code)
            if not valid:
                feedback = f"Syntax error: {err}. Viet lai."
                logger.warning(f"  Inner iter {i+1}: syntax error")
                continue
            """
            metrics = {}
            if self.engine and self.runner:
                try:
                    cls     = self.runner.load_alpha_class(code)
                    signals = self.engine.compute_signals(cls)
                    if not signals.empty:
                        metrics = self.engine.evaluate(signals, forward_days=5)
                        ic     = metrics.get("ic", 0)
                        sharpe = metrics.get("sharpe", 0)
                        wr     = metrics.get("win_rate", 0)
                        logger.info(f"  IC={ic:.4f} Sharpe={sharpe:.4f} WR={wr:.2%}")
                        feedback = (
                            f"IC={ic:.4f}, Sharpe={sharpe:.4f}, WinRate={wr:.2%}. "
                            + _suggest(ic, sharpe, wr)
                        )
                except Exception as e:
                    logger.warning(f"  Backtest loi: {e}")
                    feedback = f"Backtest loi: {str(e)[:100]}. Kiem tra lai code."
            """
            score, comment = self.judge.judge(idea, code)
            result.history.append({
                "iter": i + 1,
                "score": score,
                "comment": comment,
            })

            if score > result.final_score:
                result.final_code  = code
                result.final_latex = latex
                result.final_score = score

            if score >= self.threshold:
                break

            feedback = f"Judge: {comment}, Score: {score}"

        return result


def _suggest(ic, sharpe, wr):
    tips = []
    if abs(ic) < 0.01:
        tips.append("IC qua thap, thu them filter volume hoac rank cross-sectional")
    elif ic < 0:
        tips.append("IC am, thu dao dau signal")
    if sharpe < 0.5:
        tips.append("Sharpe thap, thu giam lookback hoac them clip(-2,2)")
    if wr < 0.50:
        tips.append("WinRate thap, thu ket hop them dieu kien xac nhan")
    if not tips:
        tips.append("Kha tot, thu tinh chinh tham so N hoac ket hop them signal")
    return " | ".join(tips)
