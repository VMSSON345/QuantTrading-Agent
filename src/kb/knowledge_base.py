import json, uuid, numpy as np
from pathlib import Path
from loguru import logger
from sentence_transformers import SentenceTransformer
from .alpha_schema import AlphaRecord, AlphaMetrics
from ..utils.paths import KB_DIR


class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"✓ Loaded embedding model: {model_name}")
        except Exception as e:
            logger.warning(f"⚠️ Fallback dummy embedder: {e}")
            self.model = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.model is None:
            return np.zeros((len(texts), 384))

        try:
            embeddings = self.model.encode(texts, show_progress_bar=False, 
                                         convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"Embedding lỗi: {e}")
            return np.zeros((len(texts), 384))

    def cosine_sim(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_norm = np.linalg.norm(a, axis=1)[:, np.newaxis]
        b_norm = np.linalg.norm(b, axis=1)
        return np.dot(a, b.T) / (a_norm * b_norm[np.newaxis, :])


class KnowledgeBase:
    def __init__(self, kb_path: str = "/workspace/thviet/quant/kb_store/kb.json"):
        self.path = Path(kb_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict = {}
        self.embedder = Embedder()
        self._idea_embeddings = np.array([])
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                for item in json.loads(self.path.read_text()):
                    d = item.copy()
                    m = AlphaMetrics(**d.pop("metrics"))
                    valid_fields = {f.name for f in AlphaRecord.__dataclass_fields__.values()}
                    d = {k: v for k, v in d.items() if k in valid_fields}
                    self._records[d["alpha_id"]] = AlphaRecord(metrics=m, **d)
                logger.info(f"✓ Loaded {len(self._records)} KB records")
            except Exception as e:
                logger.warning(f"⚠️ KB load lỗi: {e}")
        
        # Cache embeddings sau khi load
        self._cache_embeddings()

    def _cache_embeddings(self):
        if not self._records:
            self._idea_embeddings = np.array([])
            return
        
        ideas = [r.idea for r in self._records.values()]
        self._idea_embeddings = self.embedder.embed(ideas)
        logger.info(f"✓ Cached {len(self._idea_embeddings)} idea embeddings")

    def _save(self):
        rows = []
        for r in self._records.values():
            d = r.__dict__.copy()
            d["metrics"] = r.metrics.__dict__
            rows.append(d)
        self.path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    def add(self, record: AlphaRecord) -> str:
        if not record.alpha_id:
            record.alpha_id = str(uuid.uuid4())[:8]
        self._records[record.alpha_id] = record
        self._cache_embeddings()  # Recache sau khi thêm
        self._save()
        logger.info(f"✓ Added to KB: {record.name} (IC={record.metrics.ic:.4f})")
        return record.alpha_id

    def list_all(self, min_ic: float = -999.0, top_k: int = None) -> list:
        records = sorted(
            [r for r in self._records.values() if r.metrics.ic >= min_ic],
            key=lambda r: r.metrics.ic,
            reverse=True
        )
        return records[:top_k] if top_k else records

    def get_stats(self) -> dict:
        records = self.list_all()
        if not records:
            return {"total": 0, "avg_ic": 0.0, "avg_sharpe": 0.0, "best_ic": 0.0}
        
        ics    = [r.metrics.ic for r in records]
        sharps = [r.metrics.sharpe for r in records]
        return {
            "total":      len(records),
            "avg_ic":     round(sum(ics) / len(ics), 6),
            "avg_sharpe": round(sum(sharps) / len(sharps), 6),
            "best_ic":    round(max(ics), 6),
            "best_sharpe": round(max(sharps), 6),
        }

    def count(self) -> int:
        return len(self._records)

    def retrieve_similar(self, idea: str, top_k: int = 3) -> list[AlphaRecord]:
        if not self._records or len(self._idea_embeddings) == 0:
            return self.list_all()[:top_k]

        query_emb = self.embedder.embed([idea])[0]
        sims = self.embedder.cosine_sim(
            query_emb.reshape(1, -1),
            self._idea_embeddings
        )[0]

        scored = []
        for rec, sim in zip(self._records.values(), sims):
            score = sim * 0.7 + (rec.metrics.ic + 1) * 0.3
            scored.append((rec, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # FIX: trả về list[AlphaRecord], không phải list[tuple]
        return [rec for rec, _ in scored[:top_k]]