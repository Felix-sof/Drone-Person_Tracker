"""
Quick demo launcher: pick a video from a menu and open the tracker window.

Usage:
    python run_demo.py            -> choose from the menu
    python run_demo.py 2          -> open entry #2 directly
    python run_demo.py 2 --record -> also save annotated video + CSV to out/

The video list (and optionally a detector weights path) lives in
demo_videos.local.json next to this file. That file is gitignored on purpose
-- it holds machine-specific paths to your own footage. Copy
demo_videos.example.json to demo_videos.local.json and edit it. Without it,
only the webcam entry is offered.

In the window: 'h' control panel, 'p' pause, TAB select target, 'q' quit.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_CONFIG = PROJECT_DIR / "demo_videos.local.json"
EXAMPLE_CONFIG = PROJECT_DIR / "demo_videos.example.json"


def _load_local_config() -> dict:
    if not LOCAL_CONFIG.exists():
        print(f"[info] {LOCAL_CONFIG.name} not found -- copy {EXAMPLE_CONFIG.name} to "
              f"{LOCAL_CONFIG.name} and list your videos there. Offering webcam only.")
        return {}
    with open(LOCAL_CONFIG, encoding="utf-8") as f:
        return json.load(f)


def _resolve(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


def _choose(entries, argv) -> int:
    if argv and argv[0].isdigit():
        return int(argv[0]) - 1
    print("\n=== Drone Person Tracker - Demo ===")
    for i, (name, path) in enumerate(entries, 1):
        missing = "" if path is None or path.exists() else "  (file not found)"
        print(f"  {i}) {name}{missing}")
    choice = input("Select [1]: ").strip() or "1"
    return int(choice) - 1


def main():
    argv = sys.argv[1:]
    record = "--record" in argv
    argv = [a for a in argv if a != "--record"]

    local = _load_local_config()
    entries = [(v["name"], _resolve(v["path"])) for v in local.get("videos", [])]
    entries.append(("Webcam (live)", None))

    idx = _choose(entries, argv)
    if not 0 <= idx < len(entries):
        sys.exit(f"Invalid choice: {idx + 1}")
    name, video = entries[idx]
    if video is not None and not video.exists():
        sys.exit(f"Video not found: {video}")

    sys.path.insert(0, str(PROJECT_DIR))
    import config
    if local.get("model"):
        config.YOLO_MODEL = str(_resolve(local["model"]))
    if not Path(config.YOLO_MODEL).exists() and not (PROJECT_DIR / config.YOLO_MODEL).exists():
        print(f"[warn] Detector weights {config.YOLO_MODEL!r} not found -- falling back to "
              f"COCO yolov8n (auto-downloads). Set \"model\" in {LOCAL_CONFIG.name}.")
        config.YOLO_MODEL = "yolov8n.pt"
        config.PERSON_CLASS_NAMES = ["person"]
    config.ENABLE_DB_LOGGING = bool(local.get("db_logging", False))

    app_args = ["app/main.py"]
    if video is not None:
        app_args += ["--video", str(video), "--loop"]
    if record:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = PROJECT_DIR / "out"
        app_args += ["--output", str(out / f"demo_{stamp}.mp4"),
                     "--export", str(out / f"demo_{stamp}.csv")]

    print(f"\nOpening: {name}  |  model: {Path(config.YOLO_MODEL).name}")
    print("Click the window once (keyboard focus). 'h' panel, 'p' pause, 'q' quit.\n")
    sys.argv = app_args
    import app.main as tracker_app
    tracker_app.main()


if __name__ == "__main__":
    main()
