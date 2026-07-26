"""One-off patch: wires MySQL logging into pipeline.py. Safe to run once."""
import pathlib

p = pathlib.Path("src/pipeline.py")
text = p.read_text(encoding="utf-8")

edits = [
    (
        "    DISTANCE_ESTIMATION_INTERVAL,\n",
        "    DB_LOG_INTERVAL,\n    DISTANCE_ESTIMATION_INTERVAL,\n",
    ),
    (
        "from src.detection import PersonDetector\n",
        "from src.db import log_target_event\nfrom src.detection import PersonDetector\n",
    ),
    (
        "        status[\"activity\"] = posture_label if posture_label is not None else speed_activity\n\n        return status\n",
        "        status[\"activity\"] = posture_label if posture_label is not None else speed_activity\n\n"
        "        if (self._frame_count + target.id) % DB_LOG_INTERVAL == 0:\n"
        "            log_target_event(\n"
        "                target.id, status[\"activity\"], posture_label, target.last_emotion,\n"
        "                target.last_distance_m, matched_box,\n"
        "            )\n\n"
        "        return status\n",
    ),
]

for old, new in edits:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"BEKLENMEYEN: bir bolum {count} kez bulundu (1 olmali). Dosyayi elle kontrol et: {old[:50]!r}")
    text = text.replace(old, new)

p.write_text(text, encoding="utf-8")
print("Basarili: pipeline.py guncellendi.")
