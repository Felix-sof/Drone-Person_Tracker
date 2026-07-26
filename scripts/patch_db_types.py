"""One-off patch: casts numpy types to native Python types before MySQL insert."""
import pathlib

p = pathlib.Path("src/db.py")
text = p.read_text(encoding="utf-8")

old = """    try:
        bx1, by1, bx2, by2 = box
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO target_events "
            "(target_id, event_time, activity_state, posture_state, emotion, distance_m, "
            "bbox_x1, bbox_y1, bbox_x2, bbox_y2) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (target_id, datetime.now(), activity_state, posture_state, emotion, distance_m,
             bx1, by1, bx2, by2),
        )
        conn.commit()
        cursor.close()"""

new = """    try:
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
        cursor.close()"""

count = text.count(old)
if count != 1:
    raise SystemExit(f"BEKLENMEYEN: bolum {count} kez bulundu (1 olmali).")
p.write_text(text.replace(old, new), encoding="utf-8")
print("Basarili: db.py guncellendi.")
