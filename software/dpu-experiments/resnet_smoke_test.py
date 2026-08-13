import os
import numpy as np
import xir
import vart

# Optional: load overlay if running on PYNQ and bitstream exists
try:
    from pynq import Overlay
    if os.path.exists("new.bit"):
        print("Loading overlay new.bit...")
        Overlay("new.bit")
except Exception as e:
    print("Overlay load skipped or failed:", e)

# Placeholder: use ResNet50 example xmodel from Vitis AI examples
example_xmodel = "/usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel"
if not os.path.exists(example_xmodel):
    raise FileNotFoundError(
        f"Example xmodel not found at {example_xmodel}. Please set path to a valid .xmodel file."
    )

print("Using xmodel:", example_xmodel)


def np_dtype_from_tensor(tensor):
    # Convert VART tensor data type string to numpy dtype
    dt = getattr(tensor, "data_type", None)
    if dt is None:
        dt = getattr(tensor, "dtype", None)
    if dt is None:
        raise RuntimeError("Cannot infer tensor dtype")
    if isinstance(dt, str):
        dt = dt.lower()
        if dt in ("u8", "uint8"):
            return np.uint8
        if dt in ("s8", "int8"):
            return np.int8
        if dt in ("f32", "float32"):
            return np.float32
        if dt in ("f16", "float16"):
            return np.float16
        if dt in ("u32", "uint32"):
            return np.uint32
        if dt in ("s32", "int32"):
            return np.int32
        raise RuntimeError(f"Unknown tensor dtype string: {dt}")
    try:
        return np.dtype(dt)
    except Exception as ex:
        raise RuntimeError(f"Unknown tensor dtype: {dt}") from ex


# Load xmodel and create DPU runner
graph = xir.Graph.deserialize(example_xmodel)
root = graph.get_root_subgraph()
subgraphs = root.toposort_child_subgraph()
dpu_subgraphs = [s for s in subgraphs if s.get_attr("device") == "DPU"]
if not dpu_subgraphs:
    raise RuntimeError("No DPU subgraph found in xmodel")
dpu_subgraph = dpu_subgraphs[0]
runner = vart.Runner.create_runner(dpu_subgraph, "run")
print("Runner created.")

input_tensors = runner.get_input_tensors()
output_tensors = runner.get_output_tensors()

print("=== Input Tensors ===")
for i, t in enumerate(input_tensors):
    dtype = np_dtype_from_tensor(t)
    print(f"Input[{i}] name={t.name}, dims={tuple(t.dims)}, dtype={dtype}, size={np.prod(t.dims)}")

print("=== Output Tensors ===")
for i, t in enumerate(output_tensors):
    dtype = np_dtype_from_tensor(t)
    print(f"Output[{i}] name={t.name}, dims={tuple(t.dims)}, dtype={dtype}, size={np.prod(t.dims)}")

# Allocate buffers
input_bufs = []
for t in input_tensors:
    dtype = np_dtype_from_tensor(t)
    shape = tuple(t.dims)
    input_bufs.append(np.zeros(shape, dtype=dtype))

output_bufs = []
for t in output_tensors:
    dtype = np_dtype_from_tensor(t)
    shape = tuple(t.dims)
    output_bufs.append(np.zeros(shape, dtype=dtype))

# Fill first input with dummy data
if input_bufs:
    first_dtype = input_bufs[0].dtype
    if np.issubdtype(first_dtype, np.floating):
        input_bufs[0][...] = np.random.rand(*input_bufs[0].shape).astype(first_dtype)
    else:
        input_bufs[0][...] = np.random.randint(0, 255, size=input_bufs[0].shape, dtype=first_dtype)

# Run inference
job_id = runner.execute_async(input_bufs, output_bufs)
runner.wait(job_id)
print("Inference complete.")

for i, out in enumerate(output_bufs):
    print(
        f"Output buffer[{i}] shape={out.shape}, dtype={out.dtype}, min={out.min()}, max={out.max()}, mean={out.mean():.6f}"
    )

print("=== VART flow test done ===")
