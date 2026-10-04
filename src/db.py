"""
MySQL logging for target state over time. Fire-and-forget: if the DB is
unreachable, the tracker keeps running -- this must never crash or stall
the video loop.

Inserts run on a background worker thread fed by a bounded queue, so a slow
or unreachable server costs the video loop nothing (previously every
log call did a synchronous connect + INSERT + COMMIT inline, and with the
DB down, EVERY call retried the connection -- a remote host that silently
drops packets could freeze the video for the full TCP timeout each time).
If the queue fills up (DB far slower than events arrive), new events are
dropped rather than growing memory without bound. After a failed
connection, reconnects are retried at most once every
DB_RECONNECT_BACKOFF_S seconds.

mysql-connector-python is imported lazily: if it isn't installed, logging
quietly disables itself instead of breaking `import src.pipeline`.
"""
import atexit
import logging
import os
import queue
import threading
import time
from datetime import datetime

from config import DB_RECONNECT_BACKOFF_S, ENABLE_DB_LOGGING

logger = logging.getLogger(__name__)

_QUEUE_MAX = 2000
_INSERT_SQL = (
    "INSERT INTO target_events "
    "(target_id, event_time, activity_state, posture_state, emotion, distance_m, "
    "bbox_x1, bbox_y1, bbox_x2, bbox_y2) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
)
_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS target_events (
    id             BIGINT AUTO_INCREMENT PRIMARY KEY,
    target_id      INT NOT NULL,
    event_time     DATETIME(3) NOT NULL,
    activity_state VARCHAR(32),
    posture_state  VARCHAR(32),
    emotion        VARCHAR(32),
    distance_m     FLOAT NULL,
    bbox_x1 INT, bbox_y1 INT, bbox_x2 INT, bbox_y2 INT,
    INDEX idx_target_time (target_id, event_time)
)
"""


class _AsyncMySQLLogger:
    def __init__(self):
        self._queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAX)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._conn = None
        self._next_connect_attempt = 0.0
        self._disabled = not ENABLE_DB_LOGGING
        self.dropped_events = 0

    # -- producer side (video loop) ------------------------------------
    def submit(self, row: tuple) -> None:
        if self._disabled:
            return
        self._ensure_worker()
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self.dropped_events += 1

    def shutdown(self, timeout: float = 3.0) -> None:
        """Flush what's queued (bounded by `timeout`) and close."""
        thread = self._thread
        if thread is None:
            return
        try:
            self._queue.put(None, timeout=timeout)
        except queue.Full:
            pass
        thread.join(timeout=timeout)
        self._thread = None

    def _ensure_worker(self) -> None:
        if self._thread is not None:
            return
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="db-logger", daemon=True)
                self._thread.start()

    # -- consumer side (worker thread) ---------------------------------
    def _run(self) -> None:
        while True:
            row = self._queue.get()
            if row is None:
                break
            batch = [row]
            # Drain whatever else is already waiting into the same commit.
            while len(batch) < 200:
                try:
                    nxt = self._queue.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    self._write(batch)
                    self._close()
                    return
                batch.append(nxt)
            self._write(batch)
        self._close()

    def _write(self, batch: list) -> None:
        conn = self._get_connection()
        if conn is None:
            return
        try:
            cursor = conn.cursor()
            cursor.executemany(_INSERT_SQL, batch)
            conn.commit()
            cursor.close()
        except Exception as e:  # mysql.connector.Error, but module is lazy-imported
            logger.warning(f"MySQL insert basarisiz, atlaniyor: {e}")
            self._close()

    def _get_connection(self):
        if self._conn is not None:
            try:
                if self._conn.is_connected():
                    return self._conn
            except Exception:
                pass
            self._conn = None

        now = time.monotonic()
        if now < self._next_connect_attempt:
            return None
        self._next_connect_attempt = now + DB_RECONNECT_BACKOFF_S

        try:
            import mysql.connector
            from dotenv import load_dotenv
        except ImportError as e:
            logger.warning(f"MySQL istemcisi yuklu degil, DB loglama kapatildi: {e}")
            self._disabled = True
            return None

        load_dotenv()
        try:
            self._conn = mysql.connector.connect(
                host=os.getenv("MYSQL_HOST", "localhost"),
                port=int(os.getenv("MYSQL_PORT", 3306)),
                user=os.getenv("MYSQL_USER"),
                password=os.getenv("MYSQL_PASSWORD"),
                database=os.getenv("MYSQL_DATABASE"),
                connection_timeout=5,
            )
            cursor = self._conn.cursor()
            cursor.execute(_CREATE_SQL)
            self._conn.commit()
            cursor.close()
        except Exception as e:
            logger.warning(f"MySQL baglanti kurulamadi, loglama atlaniyor "
                           f"({DB_RECONNECT_BACKOFF_S}s sonra tekrar denenecek): {e}")
            self._conn = None
        return self._conn

    def _close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None


_logger = _AsyncMySQLLogger()
atexit.register(_logger.shutdown)


def log_target_event(target_id, activity_state, posture_state, emotion, distance_m, box):
    """Queue one target-state row for insertion. Never blocks, never raises."""
    if box is None:
        return
    try:
        bx1, by1, bx2, by2 = (int(v) for v in box)
        row = (
            int(target_id), datetime.now(), activity_state, posture_state, emotion,
            float(distance_m) if distance_m is not None else None,
            bx1, by1, bx2, by2,
        )
    except (TypeError, ValueError) as e:
        logger.warning(f"Gecersiz DB satiri atlandi: {e}")
        return
    _logger.submit(row)


def shutdown(timeout: float = 3.0) -> None:
    _logger.shutdown(timeout)
