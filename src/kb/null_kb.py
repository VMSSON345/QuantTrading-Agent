# src/kb/null_kb.py
class NullKB:
    """KB giả — luôn trả về rỗng, dùng để ablate main KB retrieval."""
    def retrieve_similar(self, idea, top_k=1):
        return []
    def add(self, rec):
        pass


class NullAlpha101KB:
    """Alpha101 KB giả — luôn trả về rỗng, dùng để ablate 101 retrieval."""
    def retrieve(self, idea, top_k=1):
        return []