"""
MySQL logging for target state over time. Fire-and-forget: if the DB is
unreachable, the tracker keeps running -- this must never crash or stall
the video loop.
"""
import os
import logging
from datetime import datetime

import mysql.connector
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_conn = None


def _get_connection():
    global _conn
    if _conn is None or not _conn.is_connected():
        try:
            _conn = mysql.connector.connect(
                host=os.getenv("MYSQL_HOST", "localhost"),
                port=int(os.getenv("MYSQL_PORT", 3306)),
                user=os.getenv("MYSQL_USER"),
                password=os.getenv("MYSQL_PASSWORD"),
                database=os.getenv("MYSQL_DATABASE"),
            )
        except mysql.connector.Error as e:
            logger.warning(f"MySQL baglanti kurulamadi, loglama atlaniyor: {e}")
            _conn = None
    return _conn


def log_target_event(target_id, activity_state, posture_state, emotion, distance_m, box):
    conn = _get_connection()
    if conn is None:
        return
    try:
        bx1, by1, bx2, by2 = (int(v) for v in box)
        params = (
            int(target_id), datetime.now(), activity_state, posture_state, emotion,
            float(distance_m) if distance_m is not None else None,
            bx1, by1, bx2, by2,
        )
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO target_events "
            "(target_id, event_time, activity_state, posture_state, emotion, distance_m, "
            "bbox_x1, bbox_y1, bbox_x2, bbox_y2) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            params,
        )
        conn.commit()
        cursor.close()
    except mysql.connector.Error as e:
        logger.warning(f"MySQL insert basarisiz, atlaniyor: {e}")
