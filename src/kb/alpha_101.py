"""
alpha101_kb.py
──────────────
Knowledge Base for WorldQuant 101 Alphas.
Supports:
  - Loading from JSON (id, inputs, code)
  - Semantic retrieval via sentence-transformers
  - Keyword / input-column filtering
  - Deduplication against existing AlphaRecord KB
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

# ── optional heavy dep ────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    logger.warning("sentence-transformers not installed — embeddings disabled")


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Alpha101Record:
    """One row from the WorldQuant 101 alpha dataset."""
    id: str                          # e.g. "Alpha#1"
    inputs: list[str]                # e.g. ["close", "volume"]
    code: str                        # Python expression string
    # optional enrichment added at runtime
    description: str = ""            # auto-generated plain-English summary
    tags: list[str] = field(default_factory=list)

    # numeric index for fast array lookup
    _index: int = field(default=-1, repr=False, compare=False)

    # ── helpers ───────────────────────────────────────────────────────────────

    @property
    def alpha_number(self) -> int:
        """Return 1-based integer index from id like 'Alpha#42'."""
        m = re.search(r"\d+", self.id)
        return int(m.group()) if m else -1

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "inputs": self.inputs,
            "code": self.code,
            "description": self.description,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict, index: int = -1) -> "Alpha101Record":
        rec = cls(
            id=d["id"],
            inputs=d.get("inputs", []),
            code=d.get("code", ""),
            description=d.get("description", ""),
            tags=d.get("tags", []),
        )
        rec._index = index
        return rec


# ─────────────────────────────────────────────────────────────────────────────
# Embedder  (reuses / mirrors your existing Embedder class)
# ─────────────────────────────────────────────────────────────────────────────

class Alpha101Embedder:
    DIM = 384

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = None
        if not _ST_AVAILABLE:
            return
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"✓ Alpha101Embedder loaded: {model_name}")
        except Exception as e:
            logger.warning(f"⚠️ Alpha101Embedder fallback (dummy): {e}")

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.model is None or not texts:
            return np.zeros((len(texts), self.DIM), dtype=np.float32)
        try:
            return self.model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,   # L2-normalise → dot == cosine
            ).astype(np.float32)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return np.zeros((len(texts), self.DIM), dtype=np.float32)

    @staticmethod
    def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """(N,D) × (M,D) → (N,M) cosine similarities.
        Works correctly when embeddings are L2-normalised (dot product == cosine).
        """
        # safe fallback for non-normalised embeddings
        a_n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-10
        b_n = np.linalg.norm(b, axis=1, keepdims=True) + 1e-10
        return (a / a_n) @ (b / b_n).T


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Base
# ─────────────────────────────────────────────────────────────────────────────

class Alpha101KB:
    """
    Knowledge Base for the 101 WorldQuant alphas.

    Usage
    -----
    kb = Alpha101KB("path/to/alpha101.json")

    # semantic search
    results = kb.retrieve(idea="momentum based on volume and returns", top_k=5)

    # filter by required inputs
    results = kb.filter_by_inputs(["close", "volume"], top_k=10)

    # get one alpha by id
    alpha = kb.get("Alpha#42")

    # similarity dedup against a new idea string
    similar = kb.find_similar_to_idea("correlation of returns and volume rank", top_k=3)
    """

    def __init__(
        self,
        json_path: str = "/workspace/thviet/quant/kb_store/101_alpha.json",
        model_name: str = "all-MiniLM-L6-v2",
        auto_describe: bool = True,
    ):
        self.path = Path(json_path)
        self.embedder = Alpha101Embedder(model_name)
        self._records: list[Alpha101Record] = []
        self._embeddings: np.ndarray = np.array([])   # (N, D)

        self._load(auto_describe)
        self._build_index()

    # ── I/O ───────────────────────────────────────────────────────────────────
    def extract_metadata(self, code: str) -> dict:
        # Extract window numbers từ rolling/diff/shift
        windows = []
        windows += [int(x) for x in re.findall(r'\.rolling\((\d+)\)', code)]
        windows += [int(x) for x in re.findall(r'\.diff\((\d+)\)',    code)]
        windows += [int(x) for x in re.findall(r'\.shift\((\d+)\)',   code)]

        # Extract operators
        ops = re.findall(
            r'\.(pct_change|rolling|rank|corr|diff|shift|mean|std|'
            r'apply|log|abs|sign|clip|cumsum|argmax|argmin|where)\b',
            code
        )

        return {
            "operators": list(dict.fromkeys(ops)),   # dedup, giữ thứ tự
            "windows":   sorted(set(windows)),
        }

    def _load(self, auto_describe: bool):
        if not self.path.exists():
            logger.warning(f"⚠️ Alpha101 JSON not found: {self.path}")
            return

        raw = json.loads(self.path.read_text(encoding="utf-8"))
        for i, item in enumerate(raw):
            rec = Alpha101Record.from_dict(item, index=i)
            if auto_describe and not rec.description:
                rec.description = _auto_describe(rec)
            # ── THÊM: extract metadata ──────────────────
            meta = self.extract_metadata(rec.code)
            rec.tags = meta["operators"]
            # lưu windows vào description để embed được
            if meta["windows"]:
                rec.description += f" Windows: {meta['windows']}."

            self._records.append(rec)

        logger.info(f"✓ Alpha101KB loaded {len(self._records)} alphas from {self.path}")

    def _build_index(self):
        """Embed the text representation of every alpha for semantic search."""
        if not self._records:
            return

        texts = [_record_to_text(r) for r in self._records]
        self._embeddings = self.embedder.embed(texts)
        logger.info(f"✓ Alpha101KB embeddings built: shape={self._embeddings.shape}")

    # ── public API ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._records)

    def get(self, alpha_id: str) -> Optional[Alpha101Record]:
        """Exact lookup by id, e.g. 'Alpha#7'."""
        for r in self._records:
            if r.id.lower() == alpha_id.lower():
                return r
        return None

    def get_by_number(self, n: int) -> Optional[Alpha101Record]:
        """Lookup by 1-based integer, e.g. get_by_number(7) → Alpha#7."""
        for r in self._records:
            if r.alpha_number == n:
                return r
        return None

    def list_all(self) -> list[Alpha101Record]:
        return list(self._records)

    # ── semantic retrieval ────────────────────────────────────────────────────

    def retrieve(self, idea: str, top_k: int = 1) -> list[Alpha101Record]:
        """
        Return top_k alphas most semantically similar to `idea`.
        Falls back to returning the first top_k alphas if embedder is unavailable.
        """
        if len(self._embeddings) == 0:
            logger.warning("No embeddings — returning first records")
            return self._records[:top_k]

        q_emb = self.embedder.embed([idea])                    # (1, D)
        sims = Alpha101Embedder.cosine_sim_matrix(
            q_emb, self._embeddings
        )[0]                                                    # (N,)

        top_idx = np.argsort(sims)[::-1][:top_k]
        return [self._records[i] for i in top_idx]

    def find_similar_to_idea(
        self,
        idea: str,
        top_k: int = 1,
        threshold: float = 0.75,
    ) -> list[tuple[Alpha101Record, float]]:
        """
        Return (record, similarity) pairs above `threshold`.
        Useful for deduplication checks before registering a new alpha.
        """
        if len(self._embeddings) == 0:
            return []

        q_emb = self.embedder.embed([idea])
        sims = Alpha101Embedder.cosine_sim_matrix(
            q_emb, self._embeddings
        )[0]

        results = []
        top_idx = np.argsort(sims)[::-1]
        for i in top_idx:
            s = float(sims[i])
            if s < threshold:
                break
            results.append((self._records[i], s))
            if len(results) >= top_k:
                break

        return results

    # ── filter helpers ────────────────────────────────────────────────────────

    def filter_by_inputs(
        self,
        required_inputs: list[str],
        match_all: bool = False,
        top_k: Optional[int] = None,
    ) -> list[Alpha101Record]:
        """
        Filter alphas that use any (or all) of the required_inputs columns.

        Parameters
        ----------
        required_inputs : list[str]
            Column names, e.g. ["volume", "close"].
        match_all : bool
            If True, alpha must use ALL listed inputs; default is ANY.
        top_k : int | None
            Limit output length.
        """
        req = {c.lower() for c in required_inputs}
        out = []
        for r in self._records:
            avail = {c.lower() for c in r.inputs}
            hit = req.issubset(avail) if match_all else bool(req & avail)
            if hit:
                out.append(r)
        return out[:top_k] if top_k else out

    def filter_by_keyword(
        self,
        keyword: str,
        top_k: Optional[int] = None,
    ) -> list[Alpha101Record]:
        """Case-insensitive keyword search in code + description."""
        kw = keyword.lower()
        out = [
            r for r in self._records
            if kw in r.code.lower() or kw in r.description.lower()
        ]
        return out[:top_k] if top_k else out

    # ── dedup against your main KnowledgeBase ─────────────────────────────────

    def is_duplicate_of_kb(
        self,
        alpha_idea: str,
        main_kb,                      # your existing KnowledgeBase instance
        threshold: float = 0.85,
    ) -> bool:
        """
        Return True if `alpha_idea` is too similar to something already in
        the main KnowledgeBase (uses the KB's own embedder).
        """
        try:
            similar = main_kb.retrieve_similar(alpha_idea, top_k=1)
            if not similar:
                return False
            q = main_kb.embedder.embed([alpha_idea])
            ref = main_kb.embedder.embed([similar[0].idea])
            sim = float(
                Alpha101Embedder.cosine_sim_matrix(q, ref)[0, 0]
            )
            return sim >= threshold
        except Exception as e:
            logger.warning(f"Dedup check failed: {e}")
            return False

    # ── stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        input_counter: dict[str, int] = {}
        for r in self._records:
            for inp in r.inputs:
                input_counter[inp] = input_counter.get(inp, 0) + 1

        return {
            "total_alphas": len(self._records),
            "input_usage": dict(
                sorted(input_counter.items(), key=lambda x: x[1], reverse=True)
            ),
            "embeddings_ready": len(self._embeddings) > 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _record_to_text(r: Alpha101Record) -> str:
    """
    Build the text blob that gets embedded for semantic search.
    Combines id, inputs, description, and key tokens from the code.
    """
    parts = [
        r.id,
        "inputs: " + ", ".join(r.inputs),
    ]
    if r.description:
        parts.append(r.description)
    # add code tokens (operators / function names) as lightweight bag-of-words
    tokens = re.findall(r"[a-zA-Z_]\w*", r.code)
    parts.append(" ".join(dict.fromkeys(tokens)))  # dedup, preserve order
    return " ".join(parts)


# map of operator / function name → plain-English phrase
_CODE_VOCAB = {
    "pct_change":  "percent change returns",
    "rolling":     "rolling window",
    "rank":        "cross-sectional rank",
    "corr":        "correlation",
    "diff":        "difference",
    "log":         "logarithm",
    "argmax":      "argmax position",
    "argmin":      "argmin position",
    "std":         "standard deviation",
    "mean":        "moving average",
    "sum":         "rolling sum",
    "shift":       "lag",
    "replace":     "replace values",
    "where":       "conditional selection",
    "signed_power":"signed power",
    "abs":         "absolute value",
    "sign":        "sign function",
    "clip":        "clip values",
    "apply":       "apply function",
    "fillna":      "fill missing",
    "dropna":      "drop missing",
    "cumsum":      "cumulative sum",
    "cumprod":     "cumulative product",
    "max":         "maximum",
    "min":         "minimum",
    "multiply":    "multiply",
    "divide":      "divide",
    "subtract":    "subtract",
}


def _auto_describe(r: Alpha101Record) -> str:
    """
    Generate a simple plain-English description from the code tokens.
    Not perfect, but good enough for embedding & human skimming.
    """
    tokens = re.findall(r"[a-zA-Z_]\w*", r.code)
    phrases = []
    seen: set[str] = set()
    for tok in tokens:
        if tok in _CODE_VOCAB and tok not in seen:
            phrases.append(_CODE_VOCAB[tok])
            seen.add(tok)
    inputs_str = ", ".join(r.inputs) if r.inputs else "price data"
    base = f"Alpha using {inputs_str}"
    if phrases:
        base += ": " + "; ".join(phrases[:6])
    return base + "."