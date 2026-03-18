import dataclasses
import math

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_am_bench_example(*, image_hw: int = 224) -> dict:
    """Creates a dummy input example for the am_bench adapter.

    This schema is intended for IsaacLab environments that can provide:
    - an end-effector pose (pos + quaternion)
    - a gripper width scalar
    - a single RGB image from the end-effector camera (ee_camera)

    The transforms below convert this schema into the model's expected keys:
    {state, image, image_mask, prompt}.
    """

    return {
        "am_bench/ee_pos": np.zeros((3,), dtype=np.float32),
        "am_bench/ee_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "am_bench/gripper_width": np.zeros((1,), dtype=np.float32),
        "am_bench/ee_image": np.random.randint(256, size=(image_hw, image_hw, 3), dtype=np.uint8),
        "prompt": "press the button",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _quat_wxyz_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    quat_norm = np.linalg.norm(quat)
    if quat_norm < 1e-9:
        return np.zeros((3,), dtype=np.float32)

    quat = quat / quat_norm
    scalar = float(np.clip(quat[0], -1.0, 1.0))
    sin_half_angle = math.sqrt(max(1.0 - scalar * scalar, 0.0))
    if math.isclose(sin_half_angle, 0.0):
        return np.zeros((3,), dtype=np.float32)

    angle = 2.0 * math.acos(scalar)
    axis = quat[1:] / sin_half_angle
    return (axis * angle).astype(np.float32)


@dataclasses.dataclass(frozen=True)
class AmBenchInputs(transforms.DataTransformFn):
    """Map am_bench observations into pi0/pi0.5 inputs.

    Input schema (dict keys):
    - am_bench/ee_pos: (3,) float
    - am_bench/ee_quat: (4,) float quaternion in (w, x, y, z) order
    - am_bench/gripper_width: (1,) float
    - am_bench/ee_image: uint8 (H,W,3) or float (C,H,W) / (H,W,C)
    - am_bench/base_image: optional uint8 (H,W,3) or float (C,H,W) / (H,W,C)
    - prompt: str

    Output schema (model keys):
    - state: (7,) float = [eef_pos(3), eef_axis_angle(3), gripper_width(1)]
    - image: dict with OpenPI image keys
    - image_mask: dict with masks
    - prompt: str
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        if self.model_type not in (_model.ModelType.PI0, _model.ModelType.PI05):
            raise ValueError(f"Unsupported model type for am_bench: {self.model_type}")

        ee_pos = np.asarray(data["am_bench/ee_pos"], dtype=np.float32).reshape(3)
        ee_quat = np.asarray(data["am_bench/ee_quat"], dtype=np.float32).reshape(4)
        ee_axis_angle = _quat_wxyz_to_axis_angle(ee_quat)
        gripper_width = np.asarray(data["am_bench/gripper_width"], dtype=np.float32).reshape(1)
        state = np.concatenate([ee_pos, ee_axis_angle, gripper_width], axis=0).astype(np.float32)

        ee_image = _parse_image(data["am_bench/ee_image"])
        if "am_bench/base_image" in data:
            base_image = _parse_image(data["am_bench/base_image"])
            base_mask = np.True_
        else:
            base_image = np.zeros_like(ee_image)
            base_mask = np.False_

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": ee_image,
                "right_wrist_0_rgb": np.zeros_like(ee_image),
            },
            "image_mask": {
                "base_0_rgb": base_mask,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class AmBenchOutputs(transforms.DataTransformFn):
    """Return only the first 7 action dims used by am_bench."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        return {"actions": actions[..., :7]}
