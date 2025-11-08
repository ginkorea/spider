# storage/db.py
import sqlite3
import json
import time
from pathlib import Path
from typing import Iterable, Optional, Any, List, Dict, Tuple

import numpy as np


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS pages(
  id INTEGER PRIMARY KEY,
  url TEXT UNIQUE,
  canonical TEXT,
  status INTEGER,
  fetched_at INTEGER,
  title TEXT,
  visible_text TEXT
);

CREATE TABLE IF NOT EXISTS links(
  id INTEGER PRIMARY KEY,
  from_url TEXT,
  to_url TEXT,
  anchor_text TEXT,
  rel TEXT,
  llm_score_est REAL DEFAULT 0.0,
  llm_score_final REAL DEFAULT 0.0,
  UNIQUE(from_url, to_url)
);

CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY,
  page_url TEXT,
  chunk_id INTEGER,
  text TEXT,
  token_count INTEGER,
  UNIQUE(page_url, chunk_id)
);

-- Simple vector storage (float32 array as JSON; small, portable)
CREATE TABLE IF NOT EXISTS embeddings(
  id INTEGER PRIMARY KEY,
  page_url TEXT,
  chunk_id INTEGER,
  vector TEXT,             -- json.dumps(list of floats)
  model TEXT,
  dim INTEGER,
  created_at INTEGER,
  UNIQUE(page_url, chunk_id, model)
);

CREATE TABLE IF NOT EXISTS crawl_log(
  id INTEGER PRIMARY KEY,
  url TEXT,
  action TEXT,             -- queued, fetched, skipped, failed
  reason TEXT,
  ts INTEGER
);

CREATE INDEX IF NOT EXISTS idx_pages_url ON pages(url);
CREATE INDEX IF NOT EXISTS idx_links_to ON links(to_url);
CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(page_url);
CREATE INDEX IF NOT EXISTS idx_embeds_page ON embeddings(page_url);
"""


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(a @ b / (na * nb))


class DB:
    """
    SQLite-backed storage for pages, links, chunks, and embeddings.

    New functionality:
      - has_embeddings(): quick check whether any embeddings exist (optionally by model).
      - similarity_search(): naive cosine similarity over all stored embeddings, joined with chunks.
    """

    def __init__(self, path: str = "spider_core.db"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Page / link / chunk persistence
    # ------------------------------------------------------------------
    def upsert_page(
        self,
        url: str,
        canonical: Optional[str],
        status: int,
        title: Optional[str],
        visible_text: str,
    ):
        self.conn.execute(
            """INSERT INTO pages(url, canonical, status, fetched_at, title, visible_text)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(url) DO UPDATE SET
                 canonical=excluded.canonical,
                 status=excluded.status,
                 fetched_at=excluded.fetched_at,
                 title=excluded.title,
                 visible_text=excluded.visible_text
            """,
            (url, canonical, status, int(time.time()), title, visible_text),
        )
        self.conn.commit()

    def upsert_links(self, from_url: str, links: Iterable[dict]):
        rows = []
        for l in links:
            rows.append(
                (
                    from_url,
                    l["href"],
                    l.get("text"),
                    json.dumps(l.get("rel", [])),
                    float(l.get("llm_score", 0.0)),
                )
            )
        self.conn.executemany(
            """INSERT INTO links(from_url, to_url, anchor_text, rel, llm_score_est)
               VALUES(?,?,?,?,?)
               ON CONFLICT(from_url,to_url) DO UPDATE SET
                 anchor_text=excluded.anchor_text,
                 rel=excluded.rel,
                 llm_score_est=excluded.llm_score_est
            """,
            rows,
        )
        self.conn.commit()

    def set_final_link_score(self, from_url: str, to_url: str, score: float):
        self.conn.execute(
            "UPDATE links SET llm_score_final=? WHERE from_url=? AND to_url=?",
            (float(score), from_url, to_url),
        )
        self.conn.commit()

    def upsert_chunks(self, page_url: str, chunks: Iterable[dict]):
        rows = []
        for c in chunks:
            rows.append(
                (
                    page_url,
                    int(c["chunk_id"]),
                    c["text"],
                    int(c["token_count"]),
                )
            )
        self.conn.executemany(
            """INSERT INTO chunks(page_url, chunk_id, text, token_count)
               VALUES(?,?,?,?)
               ON CONFLICT(page_url,chunk_id) DO UPDATE SET
                 text=excluded.text,
                 token_count=excluded.token_count
            """,
            rows,
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Embeddings / RAG
    # ------------------------------------------------------------------
    def upsert_embedding(
        self,
        page_url: str,
        chunk_id: int,
        vec: List[float],
        model: str,
        dim: int,
    ):
        self.conn.execute(
            """INSERT INTO embeddings(page_url,chunk_id,vector,model,dim,created_at)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(page_url,chunk_id,model) DO UPDATE SET
                 vector=excluded.vector,
                 dim=excluded.dim,
                 created_at=excluded.created_at
            """,
            (page_url, chunk_id, json.dumps(vec), model, dim, int(time.time())),
        )
        self.conn.commit()

    def has_embeddings(self, model: Optional[str] = None) -> bool:
        cur = self.conn.cursor()
        if model:
            row = cur.execute(
                "SELECT 1 FROM embeddings WHERE model=? LIMIT 1",
                (model,),
            ).fetchone()
        else:
            row = cur.execute("SELECT 1 FROM embeddings LIMIT 1").fetchone()
        return row is not None

    def similarity_search(
        self,
        query_vec: List[float],
        top_k: int = 5,
        model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Naive in-memory cosine similarity search over all embeddings
        (optionally filtered by model). Returns list of:
          {page_url, chunk_id, text, score}
        """
        cur = self.conn.cursor()
        if model:
            rows = cur.execute(
                "SELECT page_url, chunk_id, vector, dim FROM embeddings WHERE model=?",
                (model,),
            ).fetchall()
        else:
            rows = cur.execute(
                "SELECT page_url, chunk_id, vector, dim FROM embeddings",
            ).fetchall()

        if not rows:
            return []

        q = np.asarray(query_vec, dtype=np.float32)
        sims: List[Tuple[float, str, int]] = []

        for page_url, chunk_id, vec_json, dim in rows:
            try:
                v_list = json.loads(vec_json)
                v = np.asarray(v_list, dtype=np.float32)
            except Exception:
                continue
            if v.shape[0] != dim:
                continue
            score = _cosine_sim(q, v)
            sims.append((score, page_url, int(chunk_id)))

        sims.sort(key=lambda x: x[0], reverse=True)
        sims = sims[:top_k]

        # Fetch text for those chunks
        results: List[Dict[str, Any]] = []
        for score, page_url, chunk_id in sims:
            row = cur.execute(
                "SELECT text FROM chunks WHERE page_url=? AND chunk_id=?",
                (page_url, chunk_id),
            ).fetchone()
            text = row[0] if row else ""
            results.append(
                {
                    "page_url": page_url,
                    "chunk_id": chunk_id,
                    "text": text,
                    "score": float(score),
                }
            )

        return results

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def already_fetched(self, url: str) -> bool:
        r = self.conn.execute(
            "SELECT 1 FROM pages WHERE url=? LIMIT 1",
            (url,),
        ).fetchone()
        return r is not None

    def log(self, url: str, action: str, reason: Optional[str] = None):
        self.conn.execute(
            "INSERT INTO crawl_log(url,action,reason,ts) VALUES(?,?,?,?)",
            (url, action, reason, int(time.time())),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()
