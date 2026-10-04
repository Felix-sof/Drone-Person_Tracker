"""
Live/offline prototype entry point -- supports tracking MULTIPLE targets
at once.

Usage (webcam, live):
    python app/main.py --reference path/to/photo.jpg

Usage (recorded footage, e.g. drone video you downloaded):
    python app/main.py --video path/to/drone_footage.mp4 --reference path/to/photo.jpg

Several reference photos of the SAME person (front/back/side) -- the first
creates the target, the rest are added as extra gallery angles. Repeat
--reference for additional people:
    python app/main.py --video clip.mp4 --reference p1_front.jpg p1_back.jpg --reference p2.jpg

Batch / server processing (no window), saving the annotated video and a
per-frame track log plus per-target summary:
    python app/main.py --video clip.mp4 --headless --output out/clip.mp4 --export out/tracks.csv

Controls while running:
    n  -> add a NEW target from the current frame (the largest person in
          view). Refused if they already match an existing target's
          gallery closely -- that's what stops the same physical person
          being double-counted as two different targets.
    a  -> add the current frame as an ADDITIONAL angle for the
          MOST RECENTLY added/selected target (e.g. their back view after
          they turn around), without creating a new identity.
    1-9 -> select which target number 'a' applies to (defaults to the most
          recently added one).
    TAB -> cycle the selection through all current targets (also reaches
          IDs above 9).
    x  -> remove the currently selected target.
    p  -> pause / resume (video files only)
    s  -> save a snapshot of the current annotated frame (SNAPSHOT_DIR)
    r  -> start / stop recording the annotated view to recordings/
    h  -> toggle the on-screen control panel (all shortcuts + live status:
          active target count, paused state, FPS, recording, motion comp/
          tiling/distance estimation on/off)
    q  -> quit

--video accepts anything OpenCV's VideoCapture understands: a local file
path (.mp4, .avi, .mov, ...) or an RTSP/HTTP stream URL -- which is the
same mechanism you'd use to connect to a real drone's video feed later.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    CAMERA_INDEX,
    ENABLE_DISTANCE_ESTIMATION,
    ENABLE_MOTION_COMPENSATION,
    ENABLE_TILED_DETECTION,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    PROCESS_MAX_SIDE,
    ROTATE_FRAME,
    SNAPSHOT_DIR,
    VIDEO_PROCESS_EVERY_N_FRAMES,
)
from src import db
from src.config_validation import validate_config, ConfigError
from src.control_panel import ControlPanel
from src.pipeline import DronePersonTrackingPipeline
from src.recorder import TrackExporter, VideoRecorder

_ROTATE_CODES = {
    "90_CW": cv2.ROTATE_90_CLOCKWISE,
    "90_CCW": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "180": cv2.ROTATE_180,
}
_KEY_TAB = 9


def _apply_rotation(frame):
    if ROTATE_FRAME is None:
        return frame
    code = _ROTATE_CODES.get(ROTATE_FRAME)
    return cv2.rotate(frame, code) if code is not None else frame


def _limit_size(frame):
    """Downscale frames whose longer side exceeds PROCESS_MAX_SIDE."""
    if not PROCESS_MAX_SIDE:
        return frame
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= PROCESS_MAX_SIDE:
        return frame
    s = PROCESS_MAX_SIDE / longest
    return cv2.resize(frame, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


_WINDOW_NAME = "Drone Person Tracker (prototype)"
_MAX_WINDOW_HEIGHT = 850   # keep tall/vertical videos (e.g. Shorts, 9:16) from
                           # opening a window taller than the screen -- if the
                           # title bar ends up off-screen, the window can never
                           # get keyboard focus and keys silently do nothing.


def _setup_window(frame_shape):
    """Creates a resizable window sized to fit on-screen and positions it
    near the top-left corner, so its title bar is always reachable to click
    (giving it keyboard focus) regardless of the source video's aspect
    ratio."""
    h, w = frame_shape[:2]
    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
    if h > _MAX_WINDOW_HEIGHT:
        scale = _MAX_WINDOW_HEIGHT / h
        cv2.resizeWindow(_WINDOW_NAME, int(w * scale), int(h * scale))
    else:
        cv2.resizeWindow(_WINDOW_NAME, w, h)
    cv2.moveWindow(_WINDOW_NAME, 50, 50)


def _parse_args():
    parser = argparse.ArgumentParser(description="Drone person tracker")
    parser.add_argument("--reference", type=str, nargs="+", action="append", default=None,
                        metavar="IMAGE",
                        help="Reference photo(s) of ONE person to track. Extra photos after the "
                             "first are added as additional angles of the same person. Repeat "
                             "--reference to add more people.")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to a video file or stream URL (e.g. RTSP). "
                             "If omitted, falls back to the live webcam.")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX,
                        help=f"Webcam index when --video is not given (default {CAMERA_INDEX}).")
    parser.add_argument("--loop", action="store_true",
                        help="When using --video, restart from the beginning "
                             "when the video ends instead of quitting.")
    parser.add_argument("--output", type=str, default=None,
                        help="Record the annotated view to this video file (e.g. out/run.mp4).")
    parser.add_argument("--export", type=str, default=None,
                        help="Write a per-frame track log to this CSV file; a per-target "
                             "summary is written next to it as <name>.summary.json.")
    parser.add_argument("--headless", action="store_true",
                        help="No display window / keyboard (batch processing, servers). "
                             "Ctrl+C stops cleanly and still writes outputs.")
    parser.add_argument("--process-every", type=int, default=None, metavar="N",
                        help="Override VIDEO_PROCESS_EVERY_N_FRAMES (0 = analyze every frame).")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after this many source frames (quick tests).")
    return parser.parse_args()


def _load_references(pipeline, reference_groups) -> int | None:
    """Each group = photos of one person. Returns the last created target ID."""
    selected = None
    for group in reference_groups or []:
        target_id = None
        for i, path in enumerate(group):
            img = cv2.imread(path)
            if img is None:
                print(f"Could not read reference image: {path}")
                sys.exit(1)
            if i == 0 or target_id is None:
                result = pipeline.add_target(img)
                if result.ok:
                    target_id = result.target_id
            else:
                result = pipeline.add_angle(target_id, img)
            print(f"[{Path(path).name}] {result.message}")
        if target_id is not None:
            selected = target_id
    return selected


def _timestamped(directory: str, prefix: str, ext: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(directory) / f"{prefix}_{stamp}{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main():
    try:
        validate_config()
    except ConfigError as e:
        print(f"[CONFIG ERROR] {e}")
        sys.exit(1)

    args = _parse_args()
    process_every = (args.process_every if args.process_every is not None
                     else VIDEO_PROCESS_EVERY_N_FRAMES)

    pipeline = DronePersonTrackingPipeline()
    selected_target_id = _load_references(pipeline, args.reference)

    using_video_file = args.video is not None
    source = args.video if using_video_file else args.camera
    cap = cv2.VideoCapture(source)

    if not using_video_file:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"Could not open {'video' if using_video_file else 'webcam'}: {source}")
        sys.exit(1)

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    source_fps = source_fps if source_fps and source_fps > 1 else None
    # Only a real local FILE has a trustworthy frame clock; a live stream
    # (RTSP/HTTP) is effectively real time, so wall-clock time is right there.
    is_local_file = using_video_file and Path(args.video).exists()

    # For video files, pace playback to the SOURCE video's own frame rate.
    # Without this, once heavy analysis is skipped on most frames (see
    # VIDEO_PROCESS_EVERY_N_FRAMES), the loop has nothing slowing it down
    # and just reads/displays frames as fast as the CPU can go -- which
    # plays the video back FASTER than real time, not at a natural pace.
    # Headless runs skip pacing entirely: there's no one watching, so
    # finishing sooner is strictly better.
    target_frame_interval = None
    if using_video_file and not args.headless and source_fps:
        target_frame_interval = 1.0 / source_fps
    last_tick = time.monotonic()

    recorder = VideoRecorder(args.output, fps=source_fps or 25.0) if args.output else None
    exporter = TrackExporter(args.export) if args.export else None

    if not args.headless:
        print("Controls: 'n' new target, 'a' add angle to selected target, "
              "'1'-'9'/TAB select target, 'x' remove selected target"
              + (", 'p' pause/resume" if using_video_file else "")
              + ", 's' snapshot, 'r' record, 'h' control panel, 'q' quit.")

    panel = ControlPanel()
    window_ready = False
    paused = False
    refresh_requested = False
    last_frame = None
    last_annotated = None
    last_statuses = []
    skip_counter = 0
    frame_idx = 0
    processing_fps = None
    started = time.monotonic()

    try:
        while True:
            if not paused:
                if args.max_frames is not None and frame_idx >= args.max_frames:
                    print(f"Reached --max-frames={args.max_frames}.")
                    break
                ok, frame = cap.read()
                if not ok:
                    if using_video_file and args.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    print("End of stream." if using_video_file else "Camera read failed.")
                    break
                frame = _limit_size(_apply_rotation(frame))
                frame_idx += 1
                last_frame = frame

                if not window_ready and not args.headless:
                    _setup_window(frame.shape)
                    window_ready = True
            else:
                frame = last_frame

            # Video clock for files (keeps speed-based activity correct even
            # when analysis runs slower than real time); wall clock otherwise.
            timestamp = (frame_idx / source_fps) if (is_local_file and source_fps) else None

            # For video files, process_every > 0 skips the heavy pipeline on
            # most frames and just re-displays the last analyzed result,
            # trading temporal resolution for wall-clock playback speed.
            # While paused, the frame doesn't change, so it's only
            # re-analyzed when an action (new target, removal...) asks for it.
            if paused:
                do_process = refresh_requested or last_annotated is None
            else:
                do_process = (not using_video_file or skip_counter == 0
                              or last_annotated is None)
            refresh_requested = False

            if do_process:
                t0 = time.perf_counter()
                annotated, last_statuses = pipeline.process_frame(
                    frame, timestamp=timestamp, selected_id=selected_target_id)
                dt = time.perf_counter() - t0
                if dt > 0:
                    inst = 1.0 / dt
                    processing_fps = inst if processing_fps is None else 0.9 * processing_fps + 0.1 * inst
                last_annotated = annotated
                if exporter is not None and not paused:
                    exporter.add_frame(frame_idx, timestamp if timestamp is not None
                                       else time.monotonic() - started, last_statuses)
            # Draw overlays on a copy -- the cached frame is re-shown on
            # skipped/paused frames and must not accumulate text.
            annotated = last_annotated.copy()

            if using_video_file and not paused:
                skip_counter = (skip_counter + 1) % (process_every + 1)

            if paused:
                cv2.putText(annotated, "PAUSED", (20, annotated.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if recorder is not None:
                cv2.circle(annotated, (annotated.shape[1] - 20, annotated.shape[0] - 20),
                           7, (0, 0, 255), -1, cv2.LINE_AA)

            # Selection may point at a target that expired or was removed.
            if selected_target_id is not None and pipeline.get_target(selected_target_id) is None:
                selected_target_id = None

            annotated = panel.draw(annotated, state={
                "targets": len(pipeline.confirmed_targets),
                "locked": sum(1 for s in last_statuses if s.get("box") is not None),
                "selected": selected_target_id,
                "fps": processing_fps,
                "recording": recorder is not None,
                "paused": paused,
                "motion_comp": ENABLE_MOTION_COMPENSATION,
                "tiling": ENABLE_TILED_DETECTION,
                "distance": ENABLE_DISTANCE_ESTIMATION,
            })

            if recorder is not None and not paused:
                recorder.write(annotated)

            if args.headless:
                if frame_idx % 100 == 0 and do_process:
                    fps_text = f"{processing_fps:.1f}" if processing_fps else "-"
                    print(f"frame {frame_idx}: {len(pipeline.confirmed_targets)} targets, "
                          f"{fps_text} proc fps")
                continue

            cv2.imshow(_WINDOW_NAME, annotated)

            # Pace to the source video's real frame rate: figure out how much
            # time is LEFT in this frame's natural display duration after all
            # the processing above, and wait exactly that long (never less than
            # 1ms, since waitKey still needs to run for the window/keyboard to
            # be responsive; if processing overran the frame's time budget we
            # don't try to "catch up" by waiting 0 -- that would just spin).
            wait_ms = 1 if not paused else 30
            if target_frame_interval is not None and not paused:
                elapsed = time.monotonic() - last_tick
                remaining = target_frame_interval - elapsed
                if remaining > 0:
                    wait_ms = max(1, int(remaining * 1000))
            last_tick = time.monotonic()

            key = cv2.waitKey(wait_ms) & 0xFF
            # Closing the window with its X button doesn't send a key --
            # without this check the loop would keep running invisibly.
            window_closed = cv2.getWindowProperty(_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
            if key == ord("q") or window_closed:
                break

            elif key == ord("n") and last_frame is not None:
                result = pipeline.add_target(last_frame)
                print(result.message)
                if result.ok:
                    selected_target_id = result.target_id
                    refresh_requested = True

            elif key == ord("a") and last_frame is not None:
                if selected_target_id is None:
                    print("No target selected -- press 'n' to add one first.")
                else:
                    result = pipeline.add_angle(selected_target_id, last_frame)
                    print(result.message)
                    refresh_requested = True

            elif key == ord("x"):
                if selected_target_id is None:
                    print("No target selected.")
                else:
                    removed = pipeline.remove_target(selected_target_id)
                    print(f"Target #{selected_target_id} removed." if removed
                          else f"No target #{selected_target_id}.")
                    selected_target_id = None
                    refresh_requested = True

            elif ord("1") <= key <= ord("9"):
                candidate_id = key - ord("0")
                if pipeline.get_target(candidate_id) is not None:
                    selected_target_id = candidate_id
                    print(f"Selected target #{selected_target_id}.")
                    refresh_requested = True
                else:
                    print(f"No target #{candidate_id}.")

            elif key == _KEY_TAB:
                ids = sorted(t.id for t in pipeline.confirmed_targets)
                if ids:
                    later = [i for i in ids if selected_target_id is None or i > selected_target_id]
                    selected_target_id = later[0] if later else ids[0]
                    print(f"Selected target #{selected_target_id}.")
                    refresh_requested = True

            elif key == ord("p") and using_video_file:
                paused = not paused

            elif key == ord("s"):
                path = _timestamped(SNAPSHOT_DIR, f"snapshot_f{frame_idx}", ".jpg")
                cv2.imwrite(str(path), annotated)
                print(f"Snapshot saved: {path}")

            elif key == ord("r"):
                if recorder is not None:
                    recorder.close()
                    print(f"Recording stopped: {recorder.path} ({recorder.frames_written} frames)")
                    recorder = None
                else:
                    recorder = VideoRecorder(str(_timestamped("recordings", "session", ".mp4")),
                                             fps=source_fps or 25.0)
                    print(f"Recording started: {recorder.path}")

            elif key == ord("h"):
                panel.toggle()

    except KeyboardInterrupt:
        print("Interrupted -- finishing outputs...")

    finally:
        cap.release()
        if recorder is not None:
            recorder.close()
            print(f"Video saved: {recorder.path} ({recorder.frames_written} frames)")
        if exporter is not None:
            exporter.close()
            summary = exporter.summary()
            print(f"Track log saved: {exporter.csv_path} ({exporter.rows_written} rows, "
                  f"{summary['total_targets']} targets); summary: {exporter.summary_path}")
        db.shutdown()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
