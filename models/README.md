# Models

The pose model is pretrained and downloaded separately — it is not committed
here, and it was not trained by this project.

## MoveNet SinglePose (required)

`software/movenet_pose.py` expects a TFLite MoveNet SinglePose model. Default
path: `models/movenet_lightning_int8.tflite`.

| Variant | Input | Notes |
|---|---|---|
| Lightning | 192×192 | Faster — the default, and the better fit for a ZU3 PS |
| Thunder | 256×256 | More accurate, noticeably slower on ARM |

Download from Kaggle Models (formerly TF Hub):
<https://www.kaggle.com/models/google/movenet>

Save the int8 Lightning variant as `models/movenet_lightning_int8.tflite`, or
pass a different path:

```python
from movenet_pose import MoveNetPose
pose = MoveNetPose("models/movenet_thunder_fp16.tflite")
```

The module reads the input size from the model itself, so either variant works
without code changes.

MoveNet is published by Google under the Apache 2.0 license.

## Classifier weights

The workout classifier's weights are **not** here — they are baked into the
bitstream. See [`ip/mlp_controller_1_0/src/all_weights.coe`](../ip/mlp_controller_1_0/src/all_weights.coe),
which initializes the on-chip BRAM at synthesis time.
