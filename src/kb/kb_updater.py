from .knowledge_base import KnowledgeBase
from .alpha_schema import AlphaRecord
class KBUpdater:
    def __init__(self, kb: KnowledgeBase, min_ic: float = 0.01):
        self.kb = kb; self.min_ic = min_ic
    def maybe_add(self, record: AlphaRecord) -> bool:
        if record.metrics.ic >= self.min_ic:
            self.kb.add(record); return True
        return False
