"""SQLite 持久化。tag 以 JSON 数组存储，按类型/tag 过滤在 Python 侧完成（数据量小，足够简单）。"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .schema import Memory
from .taxonomy import PRIORITY_NAMES

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    memory_type TEXT NOT NULL,
    tags TEXT NOT NULL,
    priority TEXT NOT NULL,
    description TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


class MemoryStore:
    def __init__(self, path: str | Path = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(SCHEMA)

    def add(self, memory: Memory) -> Memory:
        memory.id = memory.id or uuid.uuid4().hex[:8]
        memory.created_at = memory.created_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.conn.execute(
            "INSERT OR REPLACE INTO memories VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                memory.id,
                memory.memory_type,
                json.dumps(memory.tags),
                memory.priority,
                memory.description,
                memory.content,
                memory.created_at,
            ),
        )
        self.conn.commit()
        return memory

    def all(self) -> list[Memory]:
        rows = self.conn.execute("SELECT * FROM memories ORDER BY created_at").fetchall()
        return [Memory(**(dict(r) | {"tags": json.loads(r["tags"])})) for r in rows]

    def delete(self, memory_id: str) -> bool:
        cur = self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def candidates(self, types: list[str], tags: list[str], limit: int | None = None) -> list[Memory]:
        """类型命中或任一 tag 命中即为候选；都为空时退化为全量。按优先级从高到低排序，可选截断。"""
        memories = self.all()
        if types or tags:
            memories = [m for m in memories if m.memory_type in types or set(m.tags) & set(tags)]
        memories.sort(key=lambda m: PRIORITY_NAMES.index(m.priority), reverse=True)
        return memories[:limit]
