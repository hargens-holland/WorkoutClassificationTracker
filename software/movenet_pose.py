"""
movenet_pose.py — MoveNet single-pose estimation on the PS.

Reconstructed after the capstone. The original team code ran pose estimation on
the PS but that implementation was not preserved; this is a working
implementation of the same stage, written against the public pretrained MoveNet
model.

MoveNet SinglePose returns 17 keypoints as (y, x, score), normalized to its own
square input. This module undoes the aspect-preserving letterbox so callers get
coordinates normalized to the original frame.

Model:
    MoveNet SinglePose Lightning (192x192) — fastest, used by default
    MoveNet SinglePose Thunder  (256x256) — more accurate, slower

Download the TFLite model to models/ before use:
    https://www.kaggle.com/models/google/movenet  (see models/README.md)

Runtime: tflite_runtime if present, otherwise tensorflow.lite.
"""

import numpy as np
import cv2

try:                                        # lightweight runtime, preferred on-board
    from tflite_runtime.interpreter import Interpreter
except ImportError:                         # fall back to full TensorFlow
    try:
        from tensorflow.lite import Interpreter
    except ImportError:
        Interpreter = None


# MoveNet keypoint order (COCO 17).
KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]
N_KEYPOINTS = len(KEYPOINT_NAMES)


def letterbox(frame, size):
    """
    Resize a BGR frame into a square `size`x`size` canvas, preserving aspect
    ratio and centring the image with zero padding.

    Returns:
        canvas: [size, size, 3] uint8
        box:    (top, left, height, width) of the image within the canvas
    """
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))

    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas  = np.zeros((size, size, 3), dtype=frame.dtype)
    top     = (size - nh) // 2
    left    = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, (top, left, nh, nw)


def undo_letterbox(keypoints, box, size):
    """
    Map keypoints normalized to the padded canvas back to coordinates
    normalized to the original frame.

    Args:
        keypoints: [N, 3] array of (y, x, score) in [0, 1] canvas space
        box:       (top, left, height, width) from letterbox()
        size:      canvas edge length
    Returns:
        [N, 3] array of (y, x, score); y and x in [0, 1] of the original frame.
    """
    top, left, nh, nw = box
    out = keypoints.copy()
    out[:, 0] = (keypoints[:, 0] * size - top) / nh     # y
    out[:, 1] = (keypoints[:, 1] * size - left) / nw    # x
    np.clip(out[:, :2], 0.0, 1.0, out=out[:, :2])
    return out


class MoveNetPose:
    """MoveNet SinglePose estimator running on the PS via TFLite."""

    def __init__(self, model_path="models/movenet_lightning_int8.tflite"):
        if Interpreter is None:
            raise ImportError(
                "No TFLite runtime found. Install one of:\n"
                "  pip install tflite-runtime      (recommended on the board)\n"
                "  pip install tensorflow"
            )

        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self._input  = self.interpreter.get_input_details()[0]
        self._output = self.interpreter.get_output_details()[0]

        # Input is [1, size, size, 3]; size is 192 (Lightning) or 256 (Thunder).
        self.size   = int(self._input["shape"][1])
        self._dtype = self._input["dtype"]

    def __call__(self, frame):
        """
        Estimate pose for one BGR frame.

        Args:
            frame: [H, W, 3] uint8 BGR, as returned by cv2.VideoCapture
        Returns:
            [17, 3] float32 — each row (y, x, confidence), y/x normalized to
            the input frame, confidence in [0, 1].
        """
        rgb            = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        canvas, box    = letterbox(rgb, self.size)
        tensor         = canvas[np.newaxis].astype(self._dtype)

        self.interpreter.set_tensor(self._input["index"], tensor)
        self.interpreter.invoke()

        # MoveNet SinglePose emits [1, 1, 17, 3].
        raw = self.interpreter.get_tensor(self._output["index"])
        keypoints = np.array(raw, dtype=np.float32).reshape(N_KEYPOINTS, 3)

        return undo_letterbox(keypoints, box, self.size)
