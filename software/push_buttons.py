"""
push_buttons.py — physical button UI and per-user session stats.

Runs on the PS. The four PL user push-buttons are exposed through the AXI GPIO
block in the design (see vivado/constraints/gpio_pb.xdc for pin assignment):

    BTN0  reset the current user's counts
    BTN1  start recording a session
    BTN2  end the session and commit it to the leaderboard
    BTN3  global reset (clear all users and personal bests)

Rep counting happens here on the PS, not in the PL: the classifier IP reports a
class and a done pulse, but keeps no temporal state. See docs/architecture.md.
"""

import string

import cv2

try:
    from pynq.lib.button import Button
    BOARD = True
except ImportError:
    print("[WARN] PYNQ not available — buttons disabled")
    BOARD = False

CLASS_NAMES = {
    0: "push-up",
    1: "squat",
    2: "curl",
    3: "no pose",
}

# Classes that count toward a workout tally ("no pose" does not).
TRACKED_CLASSES = [0, 1, 2]


class Stats:
    """Rep tallies for one user across the tracked exercise classes."""

    def __init__(self, name=""):
        self.name = name
        self.exercise_counts = {c: 0 for c in TRACKED_CLASSES}

    def reset_counts(self):
        self.exercise_counts = {c: 0 for c in TRACKED_CLASSES}

    def increment(self, class_idx):
        if class_idx in self.exercise_counts:
            self.exercise_counts[class_idx] += 1

    def copy(self):
        clone = Stats(self.name)
        clone.exercise_counts = dict(self.exercise_counts)
        return clone

    def __str__(self):
        parts = [f"{count} reps in {CLASS_NAMES[c]}"
                 for c, count in self.exercise_counts.items()]
        return f"{self.name}: " + ", ".join(parts)


class SessionTracker:
    """Button-driven session state plus a per-exercise leaderboard."""

    def __init__(self):
        self.current      = Stats()
        self.history      = []
        self.best         = {c: None for c in TRACKED_CLASSES}  # class -> (name, count)
        self.is_recording = False
        self.is_typing    = False
        self._typed       = ""

    # ── button actions ────────────────────────────────────────────────────────
    def reset_current(self):
        self.current.reset_counts()
        for c, holder in self.best.items():
            if holder is not None and holder[0] == self.current.name:
                self.best[c] = None

    def reset_all(self):
        self.current = Stats()
        self.history = []
        self.best    = {c: None for c in TRACKED_CLASSES}
        self.is_recording = False

    def start_session(self):
        self.is_recording = True

    def end_session(self):
        if not self.is_recording:
            return
        self.history.append(self.current.copy())
        for c in TRACKED_CLASSES:
            count = self.current.exercise_counts[c]
            if self.best[c] is None or self.best[c][1] < count:
                self.best[c] = (self.current.name, count)
        self.current      = Stats()
        self.is_recording = False

    def on_classification(self, class_idx):
        """Called once per completed rep detected by the PS rep counter."""
        if self.is_recording:
            self.current.increment(class_idx)

    # ── name entry ────────────────────────────────────────────────────────────
    def handle_key(self, key):
        """Returns False when the user asks to quit."""
        if key == 255 or key == -1:
            return True

        if self.is_typing:
            if key in (ord('\n'), ord('\r')):
                self.current.name = self._typed
                self.is_typing    = False
            elif chr(key) in string.printable:
                self._typed += chr(key)
            return True

        if key == ord('q'):
            return False
        if key == ord('n'):
            self._typed    = ""
            self.is_typing = True
        return True

    def poll_buttons(self, buttons):
        if not buttons:
            return
        if buttons[0].read():
            self.reset_current()
        if buttons[1].read():
            self.start_session()
        if buttons[2].read():
            self.end_session()
        if buttons[3].read():
            self.reset_all()

    # ── overlay ───────────────────────────────────────────────────────────────
    def draw(self, frame, origin=(0, 160), size=(300, 320)):
        x, y = origin
        w, h = size
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1)

        last = str(self.history[-1]) if self.history else "N/A"
        cv2.putText(frame, f"most recent: {last}",
                    (x + 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)

        for i, c in enumerate(TRACKED_CLASSES):
            holder = self.best[c]
            text   = f"{holder[0]} with {holder[1]}" if holder else "N/A"
            cv2.putText(frame, f"best {CLASS_NAMES[c]}: {text}",
                        (x + 10, y + (i + 2) * 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1)
        return frame

    def print_summary(self):
        print("STATS...")
        for entry in self.history:
            print(entry)


def make_buttons():
    """Return the four PL user push-buttons, or None when off-board."""
    return [Button(i) for i in range(4)] if BOARD else None
