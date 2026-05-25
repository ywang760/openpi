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


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.maximum(norm, 1.0e-9)


def _canonicalize_quat_sign(quat: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    if reference is None:
        sign = np.where(quat[..., :1] < 0.0, -1.0, 1.0)
    else:
        reference = np.asarray(reference, dtype=np.float32)
        sign = np.where(np.sum(quat * reference, axis=-1, keepdims=True) < 0.0, -1.0, 1.0)
    return quat * sign


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[..., 1:4] *= -1.0
    return quat


def _quat_mul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs = np.asarray(lhs, dtype=np.float32)
    rhs = np.asarray(rhs, dtype=np.float32)
    w1, x1, y1, z1 = np.moveaxis(lhs, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    ).astype(np.float32)


def _to_se_relative(actions: np.ndarray, state: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] < 8:
        raise ValueError(f"ee_relative requires an 8D state prefix. Got state shape {state.shape}.")
    if actions.shape[-1] < 8:
        raise ValueError(f"ee_relative requires 8D absolute EE actions. Got action shape {actions.shape}.")

    relative = actions[..., :8].copy()
    relative[..., 0:3] -= state[0:3]
    state_quat = _normalize_quat(state[3:7])
    action_quat = _canonicalize_quat_sign(_normalize_quat(relative[..., 3:7]), state_quat)
    relative[..., 3:7] = _canonicalize_quat_sign(
        _normalize_quat(_quat_mul(_quat_conjugate(state_quat), action_quat))
    )
    relative[..., 7:8] = actions[..., 7:8]
    return relative.astype(np.float32, copy=False)


def _to_se_absolute(actions: np.ndarray, state: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] < 8:
        raise ValueError(f"ee_relative requires an 8D state prefix. Got state shape {state.shape}.")
    if actions.shape[-1] < 8:
        raise ValueError(f"ee_relative requires 8D relative EE actions. Got action shape {actions.shape}.")

    absolute = actions[..., :8].copy()
    absolute[..., 0:3] += state[0:3]
    state_quat = _normalize_quat(state[3:7])
    relative_quat = _canonicalize_quat_sign(_normalize_quat(absolute[..., 3:7]))
    absolute[..., 3:7] = _canonicalize_quat_sign(
        _normalize_quat(_quat_mul(state_quat, relative_quat)),
        state_quat,
    )
    absolute[..., 7:8] = actions[..., 7:8]
    return absolute.astype(np.float32, copy=False)


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
    - delta mode state: (7,) float = [eef_pos(3), eef_axis_angle(3), gripper_width(1)]
    - ee_relative mode state: (8,) float = [eef_pos(3), eef_quat(4), gripper_width(1)]
    - image: dict with OpenPI image keys
    - image_mask: dict with masks
    - prompt: str
    """

    model_type: _model.ModelType
    action_representation: str = "delta"

    def __call__(self, data: dict) -> dict:
        if self.model_type not in (_model.ModelType.PI0, _model.ModelType.PI05):
            raise ValueError(f"Unsupported model type for am_bench: {self.model_type}")
        if self.action_representation not in ("delta", "ee_relative"):
            raise ValueError(f"Unsupported am_bench action representation: {self.action_representation}")

        ee_pos = np.asarray(data["am_bench/ee_pos"], dtype=np.float32).reshape(3)
        ee_quat = _normalize_quat(np.asarray(data["am_bench/ee_quat"], dtype=np.float32).reshape(4))
        gripper_width = np.asarray(data["am_bench/gripper_width"], dtype=np.float32).reshape(1)
        if self.action_representation == "ee_relative":
            state = np.concatenate([ee_pos, ee_quat, gripper_width], axis=0).astype(np.float32)
        else:
            ee_axis_angle = _quat_wxyz_to_axis_angle(ee_quat)
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
            actions = np.asarray(data["actions"], dtype=np.float32)
            if self.action_representation == "ee_relative":
                actions = _to_se_relative(actions, state)
            inputs["actions"] = actions.astype(np.float32, copy=False)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class AmBenchOutputs(transforms.DataTransformFn):
    """Return only the action dims used by the selected am_bench control mode."""

    action_representation: str = "delta"

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if self.action_representation == "ee_relative":
            return {"actions": _to_se_absolute(actions, data["state"])}
        if self.action_representation != "delta":
            raise ValueError(f"Unsupported am_bench action representation: {self.action_representation}")
        return {"actions": actions[..., :7]}
