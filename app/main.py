"""
Live/offline prototype entry point -- supports tracking MULTIPLE targets
at once.

Usage (webcam, live):
    python app/main.py --reference path/to/photo.jpg

Usage (recorded footage, e.g. drone video you downloaded):
    python app/main.py --video path/to/drone_footage.mp4 --reference path/to/photo.jpg

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
    x  -> remove the currently selected target.
    p  -> pause / resume (video files only)
    q  -> quit

--video accepts anything OpenCV's VideoCapture understands: a local file
path (.mp4, .avi, .mov, ...) or an RTSP/HTTP stream URL -- which is the
same mechanism you'd use to connect to a real drone's video feed later.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    ROTATE_FRAME,
    VIDEO_PROCESS_EVERY_N_FRAMES,
)
from src.config_validation import validate_config, ConfigError
from src.pipeline import DronePersonTrackingPipeline

_ROTATE_CODES = {
    "90_CW": cv2.ROTATE_90_CLOCKWISE,
    "90_CCW": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "180": cv2.ROTATE_180,
}


def _apply_rotation(frame):
    if ROTATE_FRAME is None:
        return frame
    code = _ROTATE_CODES.get(ROTATE_FRAME)
    return cv2.rotate(frame, code) if code is not None else frame


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


def main():
    try:
        validate_config()
    except ConfigError as e:
        print(f"[CONFIG ERROR] {e}")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=str, default=None,
                         help="Path to a reference photo of a person to track "
                              "(adds them as the first target)")
    parser.add_argument("--video", type=str, default=None,
                         help="Path to a video file or stream URL (e.g. RTSP). "
                              "If omitted, falls back to the live webcam.")
    parser.add_argument("--loop", action="store_true",
                         help="When using --video, restart from the beginning "
                              "when the video ends instead of quitting.")
    args = parser.parse_args()

    pipeline = DronePersonTrackingPipeline()
    selected_target_id = None

    if args.reference:
        ref_img = cv2.imread(args.reference)
        if ref_img is None:
            print(f"Could not read reference image: {args.reference}")
            sys.exit(1)
        result = pipeline.add_target(ref_img)
        print(result.message)
        if result.ok:
            selected_target_id = result.target_id

    using_video_file = args.video is not None
    source = args.video if using_video_file else CAMERA_INDEX
    cap = cv2.VideoCapture(source)

    if not using_video_file:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"Could not open {'video' if using_video_file else 'webcam'}: {source}")
        sys.exit(1)

    # For video files, pace playback to the SOURCE video's own frame rate.
    # Without this, once heavy analysis is skipped on most frames (see
    # VIDEO_PROCESS_EVERY_N_FRAMES), the loop has nothing slowing it down
    # and just reads/displays frames as fast as the CPU can go -- which
    # plays the video back FASTER than real time, not at a natural pace.
    target_frame_interval = None
    if using_video_file:
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps and source_fps > 1:
            target_frame_interval = 1.0 / source_fps
    last_tick = time.monotonic()

    print("Controls: 'n' new target, 'a' add angle to selected target, "
          "'1'-'9' select target, 'x' remove selected target"
          + (", 'p' pause/resume" if using_video_file else "") + ", 'q' quit.")

    window_ready = False
    paused = False
    last_frame = None
    last_annotated = None
    skip_counter = 0

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                if using_video_file and args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                print("End of stream." if using_video_file else "Camera read failed.")
                break
            frame = _apply_rotation(frame)
            last_frame = frame

            if not window_ready:
                _setup_window(frame.shape)
                window_ready = True
        else:
            frame = last_frame

        # For video files, VIDEO_PROCESS_EVERY_N_FRAMES > 0 skips the heavy
        # pipeline on most frames and just re-displays the last analyzed
        # result, trading temporal resolution for wall-clock playback speed.
        do_process = (
            paused
            or not using_video_file
            or skip_counter == 0
            or last_annotated is None
        )

        if do_process:
            annotated, statuses = pipeline.process_frame(frame)
            last_annotated = annotated
        else:
            annotated = last_annotated

        if using_video_file and not paused:
            skip_counter = (skip_counter + 1) % (VIDEO_PROCESS_EVERY_N_FRAMES + 1)

        if paused:
            cv2.putText(annotated, "PAUSED", (20, annotated.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(_WINDOW_NAME, annotated)

        # Pace to the source video's real frame rate: figure out how much
        # time is LEFT in this frame's natural display duration after all
        # the processing above, and wait exactly that long (never less than
        # 1ms, since waitKey still needs to run for the window/keyboard to
        # be responsive; if processing overran the frame's time budget we
        # don't try to "catch up" by waiting 0 -- that would just spin).
        wait_ms = 1
        if target_frame_interval is not None and not paused:
            elapsed = time.monotonic() - last_tick
            remaining = target_frame_interval - elapsed
            if remaining > 0:
                wait_ms = max(1, int(remaining * 1000))
        last_tick = time.monotonic()

        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord("q"):
            break

        elif key == ord("n") and last_frame is not None:
            result = pipeline.add_target(last_frame)
            print(result.message)
            if result.ok:
                selected_target_id = result.target_id

        elif key == ord("a") and last_frame is not None:
            if selected_target_id is None:
                print("No target selected -- press 'n' to add one first.")
            else:
                result = pipeline.add_angle(selected_target_id, last_frame)
                print(result.message)

        elif key == ord("x"):
            if selected_target_id is None:
                print("No target selected.")
            else:
                removed = pipeline.remove_target(selected_target_id)
                print(f"Target #{selected_target_id} removed." if removed
                      else f"No target #{selected_target_id}.")
                selected_target_id = None

        elif ord("1") <= key <= ord("9"):
            candidate_id = key - ord("0")
            if pipeline.get_target(candidate_id) is not None:
                selected_target_id = candidate_id
                print(f"Selected target #{selected_target_id}.")
            else:
                print(f"No target #{candidate_id}.")

        elif key == ord("p") and using_video_file:
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()