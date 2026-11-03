"""SQLite cache of Ollama /api/chat responses.

Key = SHA-256 of (model, messages, tools, options) as canonical JSON, so any
change to the conversation, tool schemas or options is a different call.
Each entry also stores the wall-clock seconds the original call took, so a
resumed run reports real timings instead of cache-hit timings.
"""
import hashlib
import json
import sqlite3
from pathlib import Path

CACHE_PATH = Path(__file__).resolve().parent.parent / "cache" / "responses.sqlite"


def cache_key(model, messages, tools, options):
    blob = json.dumps({"model": model, "messages": messages, "tools": tools, "options": options},
                      sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


class ResponseCache:
    def __init__(self, path=CACHE_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(path)
        self.con.execute("CREATE TABLE IF NOT EXISTS responses ("
                         "key TEXT PRIMARY KEY, model TEXT, response TEXT, elapsed_s REAL, "
                         "created TEXT DEFAULT CURRENT_TIMESTAMP)")
        self.con.commit()

    def get(self, key):
        row = self.con.execute("SELECT response, elapsed_s FROM responses WHERE key = ?",
                               (key,)).fetchone()
        return (json.loads(row[0]), row[1]) if row else None

    def put(self, key, model, response, elapsed_s):
        self.con.execute("INSERT OR REPLACE INTO responses (key, model, response, elapsed_s) "
                         "VALUES (?, ?, ?, ?)", (key, model, json.dumps(response), elapsed_s))
        self.con.commit()

    def close(self):
        self.con.close()
