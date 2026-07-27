import sqlite3
import os
from typing import Dict, Any, Optional


class DatabaseService:
    """
    Manages SQLite database storage for pending operator requests,
    making the application service stateless and ready for containerized scaling.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initializes the database schema if it does not exist."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pending_requests (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    user_query TEXT NOT NULL,
                    answer TEXT,
                    suggestion TEXT NOT NULL,
                    tg_msg_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tg_msg_id ON pending_requests(tg_msg_id)")
            conn.commit()

    def create_request(self, req_id: str, user_query: str, suggestion: str, tg_msg_id: Optional[int] = None) -> None:
        """Stores a new pending operator request."""
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO pending_requests (id, status, user_query, answer, suggestion, tg_msg_id) VALUES (?, ?, ?, ?, ?, ?)",
                (req_id, "pending", user_query, None, suggestion, tg_msg_id)
            )
            conn.commit()

    def get_request(self, req_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a request by its ID."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM pending_requests WHERE id = ?", (req_id,)).fetchone()
            if row:
                return dict(row)
            return None

    def get_request_by_tg_msg_id(self, tg_msg_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a request mapped to a Telegram message ID."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM pending_requests WHERE tg_msg_id = ?", (tg_msg_id,)).fetchone()
            if row:
                return dict(row)
            return None

    def update_request_fulfillment(self, req_id: str, answer: str) -> None:
        """Updates a request status to completed and records the verified operator answer."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE pending_requests SET status = ?, answer = ? WHERE id = ?",
                ("completed", answer, req_id)
            )
            conn.commit()
