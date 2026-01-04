import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_am_isaac_example(*, image_hw: int = 224) -> dict:
    """Creates a dummy input example for an am_isaac-style Libero adapter.

    This schema is intended for IsaacLab environments that can provide:
    - an end-effector pose (pos + axis-angle)
    - a 2-dof gripper qpos (two finger joints)
    - a single RGB image from the end-effector camera (ee_camera)

    The transforms below convert this schema into the model's expected keys:
    {state, image, image_mask, prompt}.
    """

    return {
        "am_isaac/ee_pos": np.zeros((3,), dtype=np.float32),
        "am_isaac/ee_axis_angle": np.zeros((3,), dtype=np.float32),
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


@dataclasses.dataclass(frozen=True)
class AmIsaacLiberoInputs(transforms.DataTransformFn):
    """Map IsaacLab aerial-manipulation observations into the Libero-style pi0/pi0.5 inputs.

    Input schema (dict keys):
    - am_isaac/ee_pos: (3,) float
    - am_isaac/ee_axis_angle: (3,) float axis-angle
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
        ee_axis_angle = np.asarray(data["am_isaac/ee_axis_angle"], dtype=np.float32).reshape(3)
        gripper_qpos = np.asarray(data["am_isaac/gripper_qpos"], dtype=np.float32).reshape(2)

        state = np.concatenate([ee_pos, ee_axis_angle, gripper_qpos], axis=0).astype(np.float32)

        ee_image = _parse_image(data["am_isaac/ee_image"])  # TODO: check image convention
        # FIXME: make your Isaac camera output uint8 RGB (224, 224, 3) before it hits this transform, to avoid any ambiguity about float ranges.
        
        # the raw raw image is hwc, uint8, so DON'T call get_image conversion to convert it 

        # Need to resize the image somewhere
        zeros = np.zeros_like(ee_image)

        # TODO: normalization stats might need to be re-generated

        inputs = {
            "state": state,
            "image": {
                # OpenPI expects these keys. If your environment only has an EE (eye-in-hand)
                # camera, map it to the wrist slot and pad the remaining views.
                "base_0_rgb": zeros, # TODO: might need to set something for base image
                "left_wrist_0_rgb": ee_image, 
                "right_wrist_0_rgb": zeros,
            },
            "image_mask": {
                # For PI0/PI05, mask out padded views. For PI0_FAST, keep masks True.
                "base_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "left_wrist_0_rgb": np.True_,
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

    # TODO: could add delta-to-absolute conversion here if needed

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
