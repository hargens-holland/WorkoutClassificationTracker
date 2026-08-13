"""
pulse_monitor.py — PS-side controller for the Flex-PGA workout tracker.

Runs on the ARM Cortex-A53 of a Zynq UltraScale+ ZU3EG under PYNQ.

Data path:
    USB camera (PS) -> pose estimation (PS) -> 34 keypoint features
        -> AXI-Lite registers -> MLP classifier (PL) -> class + done
        -> overlay + rep counting (PS)

The PL side is the custom `mlp_controller` IP (see hdl/). It expects exactly
34 signed 8-bit features: 17 keypoints x (y, x), normalized to [0, 1] and
scaled to a signed byte. See docs/register-map.md.

Usage:
    python3 pulse_monitor.py

Board dependencies:
    pip install pynq opencv-python-headless numpy
"""

import time

import cv2
import numpy as np

from movenet_pose import MoveNetPose, N_KEYPOINTS as MOVENET_KEYPOINTS

# ── PYNQ imports ──────────────────────────────────────────────────────────────
try:
    from pynq import Overlay
    BOARD = True
except ImportError:
    print("[WARN] PYNQ not available — running in simulation mode (no PL)")
    BOARD = False


# ── Configuration ─────────────────────────────────────────────────────────────
BITSTREAM_PATH = "workout_classifier.bit"
CAMERA_INDEX   = 0

# The PL classifier consumes 17 keypoints x 2 coords = 34 features. This is the
# same 17 MoveNet emits, which is why the two ends of the bus line up.
N_KEYPOINTS    = MOVENET_KEYPOINTS
N_FEATURES     = N_KEYPOINTS * 2

CONF_THRESHOLD = 0.1   # minimum confidence to draw / trust a keypoint

CLASS_NAMES = {
    0: "push-up",
    1: "squat",
    2: "curl",
    3: "no pose",
}

# ── AXI-Lite register map ─────────────────────────────────────────────────────
# Derived from the user logic in hdl/mlp_controller_slave_lite_v1_0_S00_AXI.v:
# slv_reg1..slv_reg34 are concatenated into the 272-bit movenet_data bus, so the
# feature words occupy 0x04..0x88. Control and status follow immediately after.
REG_DATA_BASE = 0x04                             # slv_reg1  .. slv_reg34
REG_START     = REG_DATA_BASE + N_FEATURES * 4   # slv_reg35 -> 0x8C
REG_DONE      = REG_START + 4                    # slv_reg36 -> 0x90
REG_CLASS     = REG_START + 8                    # slv_reg37 -> 0x94


# ── Pose estimation (PS) ──────────────────────────────────────────────────────
# MoveNet is a pretrained, off-the-shelf model — nothing here trains it. It maps
# a frame to 17 keypoints, which is what makes the fabric classifier small enough
# to fit. See software/movenet_pose.py.
MODEL_PATH = "models/movenet_lightning_int8.tflite"


def extract_features(keypoints):
    """
    Flatten keypoints to the 34-element feature vector the PL expects:
    [y0, x0, y1, x1, ... y16, x16]. Confidence is dropped.
    """
    features = keypoints[:, :2].flatten()
    if features.size != N_FEATURES:
        raise ValueError(
            f"PL expects {N_FEATURES} features, got {features.size}. "
            f"The classifier is built for {N_KEYPOINTS} keypoints."
        )
    return features


# ── AXI-Lite interface (PS -> PL) ─────────────────────────────────────────────
class PLInterface:
    """Writes keypoint features to the MLP controller and reads back its result."""

    def __init__(self, overlay):
        self.ip = overlay.mlp_controller_0

    def write_features(self, features):
        """Write the feature vector as signed bytes, one per 32-bit register."""
        for i, value in enumerate(features):
            scaled = int(np.clip(round(value * 127.0), -128, 127)) & 0xFF
            self.ip.write(REG_DATA_BASE + i * 4, scaled)

    def trigger_inference(self):
        """Pulse the start bit; the PL latches movenet_data_valid for one cycle."""
        self.ip.write(REG_START, 1)
        self.ip.write(REG_START, 0)

    def read_result(self, timeout_s=0.05):
        """
        Poll the done register until the PL reports a finished inference.

        Returns:
            int class index, or None if the PL did not complete in time.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.ip.read(REG_DONE):
                return self.ip.read(REG_CLASS) & 0b11
        return None


# ── Display ───────────────────────────────────────────────────────────────────
# MoveNet-ordered skeleton edges (17 keypoints).
EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),            # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # arms
    (5, 11), (6, 12), (11, 12),                # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
]


def draw_display(frame, keypoints, exercise, reps, fps):
    h, w = frame.shape[:2]

    for a, b in EDGES:
        if keypoints[a, 2] > CONF_THRESHOLD and keypoints[b, 2] > CONF_THRESHOLD:
            y1, x1 = int(keypoints[a, 0] * h), int(keypoints[a, 1] * w)
            y2, x2 = int(keypoints[b, 0] * h), int(keypoints[b, 1] * w)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for k in range(N_KEYPOINTS):
        if keypoints[k, 2] > CONF_THRESHOLD:
            y, x = int(keypoints[k, 0] * h), int(keypoints[k, 1] * w)
            cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)

    cv2.rectangle(frame, (0, 0), (300, 160), (0, 0, 0), -1)
    cv2.putText(frame, exercise.upper(),
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, f"REPS: {reps}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    return frame


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    pl = None
    if BOARD:
        print(f"Loading bitstream: {BITSTREAM_PATH}")
        pl = PLInterface(Overlay(BITSTREAM_PATH))
    else:
        print("Simulation mode — PL classifier not driven")

    print(f"Loading pose model: {MODEL_PATH}")
    estimate_pose = MoveNetPose(MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: cannot open camera")
        return

    print("Running — press Q to quit")

    prev_time = time.time()
    fps       = 0.0
    reps      = 0
    exercise  = "none"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = estimate_pose(frame)

        if pl is not None:
            pl.write_features(extract_features(keypoints))
            pl.trigger_inference()
            class_idx = pl.read_result()
            if class_idx is not None:
                exercise = CLASS_NAMES.get(class_idx, "unknown")

        now       = time.time()
        fps       = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        cv2.imshow("Pulse", draw_display(frame, keypoints, exercise, reps, fps))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
