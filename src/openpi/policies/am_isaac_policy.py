import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_am_isaac_example(*, image_hw: int = 224) -> dict:
    """Creates a dummy input example for an am_isaac-style Libero adapter.

    This schema is intended for IsaacLab environments that can provide:
    - an end-effector pose (pos + quat)
    - a 2-dof gripper qpos (two finger joints)
    - a single RGB image from the end-effector camera (ee_camera)

    The transforms below convert this schema into the model's expected keys:
    {state, image, image_mask, prompt}.
    """

    return {
        "am_isaac/ee_pos": np.zeros((3,), dtype=np.float32),
        "am_isaac/ee_quat": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # wxyz
        "am_isaac/gripper_qpos": np.zeros((2,), dtype=np.float32),
        "am_isaac/ee_image": np.random.randint(256, size=(image_hw, image_hw, 3), dtype=np.uint8),
        "prompt": "press the button",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _axis_angle_from_quat_wxyz(quat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    if quat.shape != (4,):
        raise ValueError(f"Expected quat shape (4,), got {quat.shape}.")
    if quat[0] < 0:
        quat = -quat
    qw, qx, qy, qz = quat
    mag = float(np.linalg.norm([qx, qy, qz]))
    half_angle = float(np.arctan2(mag, qw))
    angle = 2.0 * half_angle
    if abs(angle) < eps:
        denom = 0.5 - (angle * angle) / 48.0
    else:
        denom = np.sin(half_angle) / angle
    return np.asarray([qx, qy, qz], dtype=np.float32) / float(denom)


@dataclasses.dataclass(frozen=True)
class AmIsaacLiberoInputs(transforms.DataTransformFn):
    """Map IsaacLab aerial-manipulation observations into the Libero-style pi0/pi0.5 inputs.

    Input schema (dict keys):
    - am_isaac/ee_pos: (3,) float
    - am_isaac/ee_quat: (4,) float quaternion in (w,x,y,z)
    - am_isaac/gripper_qpos: (2,) float (two finger joints)
    - am_isaac/ee_image: uint8 (H,W,3) or float (C,H,W) / (H,W,C)
    - prompt: str

    Output schema (model keys):
    - state: (8,) float = [eef_pos(3), eef_axis_angle(3), gripper_qpos(2)]
    - image: dict with OpenPI image keys
    - image_mask: dict with masks
    - prompt: str
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        ee_pos = np.asarray(data["am_isaac/ee_pos"], dtype=np.float32).reshape(3)
        ee_quat = np.asarray(data["am_isaac/ee_quat"], dtype=np.float32).reshape(4)
        gripper_qpos = np.asarray(data["am_isaac/gripper_qpos"], dtype=np.float32).reshape(2)

        ee_axis_angle = _axis_angle_from_quat_wxyz(ee_quat)
        state = np.concatenate([ee_pos, ee_axis_angle, gripper_qpos], axis=0).astype(np.float32)

        ee_image = _parse_image(data["am_isaac/ee_image"])
        zeros = np.zeros_like(ee_image)

        inputs = {
            "state": state,
            "image": {
                # OpenPI expects these keys. If your environment only has an EE camera,
                # we map it to the third-person slot and pad wrist views with zeros.
                "base_0_rgb": ee_image,
                "left_wrist_0_rgb": zeros,
                "right_wrist_0_rgb": zeros,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                # For PI0/PI05 we mask out padding images. For PI0_FAST we do not mask.
                "left_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AmIsaacLiberoOutputs(transforms.DataTransformFn):
    """Return only the first 7 action dims (LIBERO convention)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
