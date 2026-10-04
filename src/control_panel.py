"""
src/control_panel.py

On-screen "operator console" overlay: a tactical/HUD-style panel listing
every keyboard shortcut plus live system status (active target count,
paused state, motion comp / tiling / distance estimation on-off), toggled
with a single key so it doesn't clutter the view by default.

No extra dependency (cv2 only, already used everywhere in this project).
English-only labels on purpose -- cv2.putText's built-in Hershey fonts
don't render non-ASCII characters (Turkish i/s/g-breve etc.) cleanly, so
keeping this panel ASCII keeps it crisp at small sizes.

--- Integration in app/main.py ---

1) Import:
     from src.control_panel import ControlPanel

2) Before the main loop:
     panel = ControlPanel()

3) In the key-handling block:
     elif key == ord('h'):
         panel.toggle()

4) Right before cv2.imshow (so the panel draws on top of everything else):
     frame = panel.draw(frame, state={
         "targets": len(pipeline.confirmed_targets),
         "locked": number_currently_visible,
         "selected": selected_target_id,
         "fps": processing_fps,
         "recording": recorder is not None,
         "paused": paused,
         "motion_comp": ENABLE_MOTION_COMPENSATION,
         "tiling": ENABLE_TILED_DETECTION,
         "distance": ENABLE_DISTANCE_ESTIMATION,
     })

All state dict keys are optional -- omitting one just shows it as STANDBY
rather than crashing.
"""

import cv2

# --- Tactical color palette (BGR, since OpenCV) ---
_BG = (6, 10, 6)
_BORDER = (40, 220, 90)
_HEADER = (90, 255, 140)
_LABEL = (60, 150, 85)
_VALUE = (200, 245, 210)
_KEYCAP = (30, 60, 30)
_KEYCAP_TEXT = (110, 255, 150)
_DOT_ON = (70, 235, 90)
_DOT_OFF = (55, 70, 210)
_DIVIDER = (35, 90, 50)


class ControlPanel:
    # (key, description) -- single source of truth; add a shortcut here and
    # it shows up in the panel automatically.
    SHORTCUTS = [
        ("N",   "ACQUIRE NEW TARGET"),
        ("A",   "ADD ANGLE TO SELECTED"),
        ("1-9", "SELECT TARGET"),
        ("TAB", "CYCLE SELECTED TARGET"),
        ("X",   "DROP SELECTED TARGET"),
        ("P",   "PAUSE / RESUME (VIDEO)"),
        ("S",   "SAVE SNAPSHOT"),
        ("R",   "START / STOP RECORDING"),
        ("H",   "TOGGLE THIS PANEL"),
        ("Q",   "TERMINATE SESSION"),
    ]

    def __init__(self, visible: bool = False):
        self.visible = visible

    def toggle(self) -> None:
        self.visible = not self.visible

    # ------------------------------------------------------------------
    def draw(self, frame, state: dict | None = None):
        """Returns `frame` untouched (zero cost) when the panel is hidden."""
        if not self.visible:
            return frame

        state = state or {}
        h, w = frame.shape[:2]

        panel_w = 300
        x0, y0 = w - panel_w - 14, 14
        x0 = max(x0, 0)

        rows = self._status_rows(state)
        header_h = 26
        section_gap = 10
        cmd_h = len(self.SHORTCUTS) * 20 + section_gap
        status_h = len(rows) * 20 + section_gap
        footer_h = 18
        panel_h = header_h + cmd_h + status_h + footer_h + 18

        frame = self._panel_background(frame, x0, y0, panel_w, panel_h)

        y = y0 + 20
        cv2.putText(frame, "TARGETING SYS // OPERATOR PANEL", (x0 + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, _HEADER, 1, cv2.LINE_AA)
        cv2.circle(frame, (x0 + panel_w - 14, y - 5), 4, _DOT_ON, -1, cv2.LINE_AA)
        y += 6
        cv2.line(frame, (x0 + 10, y), (x0 + panel_w - 10, y), _DIVIDER, 1, cv2.LINE_AA)

        y += 18
        cv2.putText(frame, "COMMANDS", (x0 + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, _HEADER, 1, cv2.LINE_AA)
        y += 6
        for key, desc in self.SHORTCUTS:
            y += 16
            self._draw_keycap(frame, x0 + 10, y, key)
            cv2.putText(frame, desc, (x0 + 52, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, _VALUE, 1, cv2.LINE_AA)

        y += section_gap
        cv2.line(frame, (x0 + 10, y), (x0 + panel_w - 10, y), _DIVIDER, 1, cv2.LINE_AA)
        y += 18
        cv2.putText(frame, "SYSTEM STATUS", (x0 + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, _HEADER, 1, cv2.LINE_AA)
        y += 6
        for label, value, is_active in rows:
            y += 16
            dot_color = _DOT_ON if is_active else _DOT_OFF
            cv2.circle(frame, (x0 + 15, y - 4), 3, dot_color, -1, cv2.LINE_AA)
            cv2.putText(frame, label, (x0 + 26, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, _LABEL, 1, cv2.LINE_AA)
            (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.putText(frame, value, (x0 + 26 + lw + 8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, _VALUE, 1, cv2.LINE_AA)

        y += section_gap
        cv2.line(frame, (x0 + 10, y), (x0 + panel_w - 10, y), _DIVIDER, 1, cv2.LINE_AA)
        cv2.putText(frame, "[H] HIDE PANEL", (x0 + 10, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, _LABEL, 1, cv2.LINE_AA)

        return frame

    # ------------------------------------------------------------------
    @staticmethod
    def _status_rows(state: dict):
        targets = state.get("targets")
        locked = state.get("locked")
        if targets is None:
            targets_text = "-"
        elif locked is None:
            targets_text = str(targets)
        else:
            targets_text = f"{locked} / {targets}"
        selected = state.get("selected")
        fps = state.get("fps")
        rows = [
            ("TARGETS LOCKED", targets_text, bool(locked if locked is not None else targets)),
            ("SELECTED", f"T{selected:02d}" if selected is not None else "NONE", selected is not None),
            ("SESSION", "PAUSED" if state.get("paused") else "ACTIVE", not state.get("paused")),
            ("PROC FPS", f"{fps:.1f}" if fps is not None else "-", fps is not None and fps >= 5),
            ("RECORDING", "REC" if state.get("recording") else "OFF", bool(state.get("recording"))),
            ("MOTION COMP", "ACTIVE" if state.get("motion_comp") else "STANDBY", bool(state.get("motion_comp"))),
            ("TILING", "ACTIVE" if state.get("tiling") else "STANDBY", bool(state.get("tiling"))),
            ("RANGE EST.", "ACTIVE" if state.get("distance") else "STANDBY", bool(state.get("distance"))),
        ]
        return rows

    @staticmethod
    def _draw_keycap(frame, x, y, key: str):
        """Small filled tag around a key name, e.g. a boxed 'N' -- reads as
        a physical keycap rather than plain text."""
        (tw, th), _ = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
        pad_x, pad_y = 5, 3
        x1, y1 = x, y - th - pad_y
        x2, y2 = x + tw + pad_x * 2, y + pad_y
        cv2.rectangle(frame, (x1, y1), (x2, y2), _KEYCAP, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), _BORDER, 1, cv2.LINE_AA)
        cv2.putText(frame, key, (x1 + pad_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, _KEYCAP_TEXT, 1, cv2.LINE_AA)

    @staticmethod
    def _panel_background(frame, x0, y0, w, h, bracket_len=14):
        """Semi-transparent dark panel with a thin border and viewfinder-
        style corner brackets, matching the tracking-box aesthetic used
        elsewhere in the app (see pipeline.py's corner-bracket boxes)."""
        fh, fw = frame.shape[:2]
        rx1, ry1, rx2, ry2 = max(0, x0), max(0, y0), min(fw, x0 + w), min(fh, y0 + h)
        if rx2 > rx1 and ry2 > ry1:
            roi = frame[ry1:ry2, rx1:rx2]
            fill = roi.copy()
            fill[:] = _BG
            cv2.addWeighted(fill, 0.78, roi, 0.22, 0, dst=roi)

        cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), _BORDER, 1, cv2.LINE_AA)

        for (cx, cy), (dx, dy) in [
            ((x0, y0), (1, 1)), ((x0 + w, y0), (-1, 1)),
            ((x0, y0 + h), (1, -1)), ((x0 + w, y0 + h), (-1, -1)),
        ]:
            cv2.line(frame, (cx, cy), (cx + dx * bracket_len, cy), _HEADER, 2, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy * bracket_len), _HEADER, 2, cv2.LINE_AA)

        return frame
