"""
inference_runner.py
-------------------
Reusable VART inference wrapper for PYNQ + DPUCZDX8G.

Phase 1/2: ResNet smoke test (confirm DPU is alive)
Phase 3:   Drop-in for MoveNet — change xmodel path and reshape dims

PS side only — PL handles resize/normalize internally via preprocessing IP.
"""

import os
import numpy as np
import xir
import vart


# ── Dtype helper ──────────────────────────────────────────────────────────────

def _np_dtype(tensor):
    """Resolve VART tensor dtype to numpy dtype."""
    dt = getattr(tensor, "data_type", None) or getattr(tensor, "dtype", None)
    if dt is None:
        raise RuntimeError("Cannot infer tensor dtype")
    if isinstance(dt, str):
        mapping = {
            "u8": np.uint8, "uint8": np.uint8,
            "s8": np.int8,  "int8":  np.int8,
            "f32": np.float32, "float32": np.float32,
            "f16": np.float16, "float16": np.float16,
        }
        key = dt.lower()
        if key in mapping:
            return mapping[key]
        raise RuntimeError(f"Unknown dtype string: {dt}")
    return np.dtype(dt)


# ── Runner ────────────────────────────────────────────────────────────────────

def load_runner(xmodel_path: str):
    """
    Load xmodel and create VART DPU runner.

    Args:
        xmodel_path: path to .xmodel file (ResNet for now, MoveNet later)
    Returns:
        vart.Runner
    """
    if not os.path.exists(xmodel_path):
        raise FileNotFoundError(f"xmodel not found: {xmodel_path}")

    graph = xir.Graph.deserialize(xmodel_path)
    subgraphs = graph.get_root_subgraph().toposort_child_subgraph()
    dpu_subgraphs = [s for s in subgraphs if s.get_attr("device") == "DPU"]
    if not dpu_subgraphs:
        raise RuntimeError("No DPU subgraph found in xmodel")

    runner = vart.Runner.create_runner(dpu_subgraphs[0], "run")
    return runner


def inspect_tensors(runner):
    """Print input/output tensor names, shapes, dtypes. Call once after load_runner."""
    print("=== Input Tensors ===")
    for i, t in enumerate(runner.get_input_tensors()):
        print(f"  [{i}] {t.name}  dims={tuple(t.dims)}  dtype={_np_dtype(t)}")
    print("=== Output Tensors ===")
    for i, t in enumerate(runner.get_output_tensors()):
        print(f"  [{i}] {t.name}  dims={tuple(t.dims)}  dtype={_np_dtype(t)}")


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(runner, frame: np.ndarray) -> list:
    """
    Run one inference pass on the DPU.

    PL preprocessing IP handles resize/normalize before DPU — this function
    just allocates buffers, copies the raw frame, and fires execute_async.

    Args:
        runner: vart.Runner from load_runner()
        frame:  np.ndarray — raw camera frame written into a pynq.allocate buffer.
                Shape must match DPU input tensor dims (PL preprocessing handles
                the actual resize; pass the pre-allocated input buffer directly).
    Returns:
        list of np.ndarray — one per output tensor (raw, not dequantized)
    """
    input_tensors  = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()

    input_bufs  = [np.zeros(tuple(t.dims), dtype=_np_dtype(t)) for t in input_tensors]
    output_bufs = [np.zeros(tuple(t.dims), dtype=_np_dtype(t)) for t in output_tensors]

    # Copy frame data into input buffer (shape must match tensor dims)
    input_bufs[0][...] = frame

    job_id = runner.execute_async(input_bufs, output_bufs)
    runner.wait(job_id)

    return output_bufs


# ── Output helpers ────────────────────────────────────────────────────────────

def top5_resnet(output_bufs: list):
    """
    Print top-5 class predictions from ResNet output.
    Confirms DPU is alive and returning non-garbage output.
    """
    raw = output_bufs[0].flatten().astype(np.float32)
    top5 = np.argsort(raw)[::-1][:5]
    print("Top-5 predictions:")
    for rank, idx in enumerate(top5):
        print(f"  {rank+1}. class {idx:4d}  score {raw[idx]:.4f}")


def parse_keypoints(output_bufs: list, output_scale: float = 1.0) -> np.ndarray:
    """
    Parse MoveNet output into [17, 3] keypoint array.
    Adjust reshape dims once Person A confirms output tensor shape.

    Args:
        output_bufs:   raw output from run_inference()
        output_scale:  dequantization scale (1.0 if float32, fixpos-derived if int8)
    Returns:
        keypoints: np.ndarray [17, 3], each row = [y, x, confidence], values in [0, 1]
    """
    raw = output_bufs[0].astype(np.float32) * output_scale
    # MoveNet output is typically [1, 1, 17, 3] — flatten to [17, 3]
    keypoints = raw.reshape(17, 3)
    return keypoints
