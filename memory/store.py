from __future__ import annotations
import json, sqlite3, uuid
from datetime import datetime
from zoneinfo import ZoneInfo
from dataclasses import dataclass



@dataclass
class Message:
    role: str
    content: str
    timestamp: str
    total_tokens:int

def conn(db_path):
    conn = sqlite3.connect(db_path,check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TEXT,
            summary TEXT DEFAULT '',
            total_planning_tokens INTEGER DEFAULT 0,
            total_answer_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TEXT,
            total_tokens INTEGER DEFAULT 0,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );

        CREATE TABLE IF NOT EXISTS turns (
            turn_id TEXT PRIMARY KEY,
            session_id TEXT,
            query TEXT,
            search_queries TEXT,
            urls_opened TEXT,
            snippets_selected TEXT,
            final_answer TEXT,
            timestamp TEXT,
            planning_tokens INTEGER DEFAULT 0,
            answer_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );
    """)
    conn.commit()
    return conn 

def new_session(conn) -> str:
    sid = str(uuid.uuid4())
    conn.execute("INSERT INTO sessions (session_id,created_at,summary) VALUES (?,?,?)",
                    (sid, datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(), ""))
    conn.commit()
    return sid

def add_message(conn, sid: str, role: str, content: str, total_tokens: int) -> None:
    conn.execute(
        """
        INSERT INTO messages (session_id, role, content, timestamp, total_tokens)
        VALUES (?, ?, ?, ?, ?)
        """,
        (sid, role, content, datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(), total_tokens)
    )
    conn.commit()

def save_turn(conn,sid, query, search_queries, urls_opened, snippets_selected, final_answer,planning_token,answer_token,total_token):
    conn.execute("INSERT INTO turns VALUES (?,?,?,?,?,?,?,?,?,?,?)",
              (str(uuid.uuid4()), sid, query, json.dumps(search_queries),
               json.dumps(urls_opened), json.dumps(snippets_selected),
               final_answer, datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),planning_token,answer_token,total_token))
    conn.commit()

def get_messages(conn,sid: str) -> list[Message]:
    rows = conn.execute(
        "SELECT role,content,timestamp,total_tokens FROM messages WHERE session_id=? ORDER BY id", (sid,)
    ).fetchall()
    return [Message(**dict(r)) for r in rows]

def update_summary(conn,sid: str, summary: str):
    conn.execute("UPDATE sessions SET summary=? WHERE session_id=?", (summary, sid))
    conn.commit()


def get_summary(conn,sid: str) -> str:
    row = conn.execute("SELECT summary FROM sessions WHERE session_id=?", (sid,)).fetchone()
    return row["summary"] if row else ""

def update_session_tokens(conn,sid: str,planning_tokens: int,answer_tokens: int,total_tokens: int):
    conn.execute(
        """
        UPDATE sessions
        SET
            total_planning_tokens = total_planning_tokens + ?,
            total_answer_tokens = total_answer_tokens + ?,
            total_tokens = total_tokens + ?
        WHERE session_id = ?
        """,
        (
            planning_tokens,
            answer_tokens,
            total_tokens,
            sid
        )
    )

    conn.commit()
