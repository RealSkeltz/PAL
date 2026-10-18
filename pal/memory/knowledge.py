"""Semantic memory: domain knowledge loaded at the start of a run, searchable like recall.

Reference documents (site procedures, manuals, building plans) are split into
chunks of about a paragraph, each carrying its heading path so it still makes
sense on its own ("Fire safety > Evacuation: ..."). Chunks get the same keyword
plus vector search as session recall, in their own index: they are a different
corpus, and are shown by source and section instead of by time.

Embeddings are cached by chunk content, so restarting with unchanged documents
doesn't embed them again.
"""
import hashlib
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid5, NAMESPACE_URL

import numpy as np

from pal.config import KNOWLEDGE_CACHE
from pal.llm.embedding_client import EmbeddingClient
from pal.memory.recall import Entry, Hit, RecallIndex

logger = logging.getLogger(__name__)

SUFFIXES = {".md", ".txt"}
CHUNK_WORDS = 120  # paragraphs are merged up to about this many words


@dataclass(frozen=True)
class Chunk:
    id: UUID
    source: str   # file name
    section: str  # heading path, e.g. "Fire safety > Evacuation"
    text: str

    @property
    def searchable(self) -> str:
        return f"{self.section}: {self.text}" if self.section else self.text


class Knowledge:
    def __init__(self, embedder: EmbeddingClient | None, cache_dir: Path = KNOWLEDGE_CACHE):
        self.embedder = embedder
        self.cache_dir = cache_dir
        self.index = RecallIndex(embedder)
        self.chunks: dict[UUID, Chunk] = {}
        self.ready = threading.Event()  # set once the initial load is done

    def load(self, paths: list[Path]):
        """Read, chunk, embed (or take from cache) and index every .md/.txt file under `paths`."""
        start = time.time()
        files = [f for p in paths for f in _files(Path(p))]
        chunks = [c for f in files for c in chunk_document(f.read_text(errors="replace"), f.name)]
        if chunks:
            vectors, embedded = self._vectors([c.searchable for c in chunks])
            self.index.add([Entry(c.id, None, "knowledge", c.searchable) for c in chunks], vectors)
            self.chunks.update((c.id, c) for c in chunks)
            logger.info(f"knowledge: {len(chunks)} chunks from {len(files)} files "
                        f"({embedded} embedded, {len(chunks) - embedded} from cache, {time.time() - start:.1f}s)")
        self.ready.set()

    def search(self, query: str, k: int = 3) -> list[Hit]:
        if not self.ready.is_set():
            return []  # still loading; nothing rather than a half-built index
        return self.index.search(query, k=k)

    def format(self, hits: list[Hit]) -> str:
        if not hits:
            return "Nothing relevant found in the reference material."
        lines = []
        for h in hits:
            c = self.chunks[h.entry.id]
            lines.append(f"[{c.source}{' > ' + c.section if c.section else ''}] {c.text}")
        return "\n".join(lines)

    def lookup_text(self, query: str, k: int = 4) -> str:
        return self.format(self.search(query, k=k))

    def _vectors(self, texts: list[str]) -> tuple[np.ndarray | None, int]:
        """Vectors for `texts`, from the cache where possible. Returns (vectors, how many were embedded)."""
        if self.embedder is None:
            return None, 0
        path = self.cache_dir / f"{re.sub(r'[^A-Za-z0-9._-]', '_', self.embedder.model)}.npz"
        cache = dict(np.load(path)) if path.exists() else {}
        keys = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
        missing = [i for i, key in enumerate(keys) if key not in cache]
        if missing:
            for start in range(0, len(missing), 64):
                batch = missing[start:start + 64]
                for i, v in zip(batch, self.embedder.vectorise([texts[i] for i in batch])):
                    cache[keys[i]] = v
            np.savez(path, **cache)
        return np.stack([cache[key] for key in keys]), len(missing)


def chunk_document(text: str, source: str) -> list[Chunk]:
    """Split on headings and blank lines; merge paragraphs under one heading up to CHUNK_WORDS.
    Every heading starts a new chunk, so a chunk never mixes sections."""
    chunks, headings, buffer, paragraph = [], [], [], []

    def end_paragraph():
        if paragraph:
            for piece in _split_long(" ".join(paragraph)):
                if buffer and len(" ".join(buffer).split()) + len(piece.split()) > CHUNK_WORDS:
                    end_chunk()
                buffer.append(piece)
            paragraph.clear()

    def end_chunk():
        if buffer:
            body = "\n".join(buffer).strip()
            section = " > ".join(h for h in headings if h)
            chunks.append(Chunk(uuid5(NAMESPACE_URL, f"{source}|{section}|{body}"), source, section, body))
            buffer.clear()

    for line in text.splitlines():
        heading = re.match(r"^(#{1,6})\s+(.*)", line.strip())
        if heading:
            end_paragraph()
            end_chunk()
            level = len(heading.group(1))
            headings[:] = (headings + [""] * level)[:level - 1] + [heading.group(2).strip()]
        elif line.strip():
            paragraph.append(line.strip())
        else:
            end_paragraph()
    end_paragraph()
    end_chunk()
    return chunks


def _split_long(block: str) -> list[str]:
    """A paragraph longer than CHUNK_WORDS is cut at sentence ends."""
    if len(block.split()) <= CHUNK_WORDS:
        return [block]
    pieces, current = [], []
    for sentence in re.split(r"(?<=[.!?])\s+", block):
        if current and len(" ".join(current).split()) + len(sentence.split()) > CHUNK_WORDS:
            pieces.append(" ".join(current))
            current = []
        current.append(sentence)
    return pieces + ([" ".join(current)] if current else [])


def _files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUFFIXES else []
    if path.is_dir():
        return sorted(f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in SUFFIXES)
    logger.warning(f"knowledge path not found: {path}")
    return []
