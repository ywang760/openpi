"""Evaluate OpenPI policies (e.g. pi05_libero) zero-shot on Isaac Lab tasks.

This script mirrors the structure of am_isaac_il/policies/act/eval.py:
- Launch Isaac Sim via AppLauncher
- Create an IsaacLab gym environment
- Run rollouts while querying an OpenPI policy for chunked actions

Notes
- For aerial-manipulation tasks in am_isaac, the env action is 7D:
  [x, y, z, roll, pitch, yaw, gripper]
- OpenPI LIBERO policies output 7D *delta* actions (pos delta + axis-angle delta + gripper).
  We convert them into absolute pose targets relative to the state at query time.
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate OpenPI policies zero-shot on Isaac Lab environments.")
parser.add_argument("--task", type=str, required=True, help="Name of the Isaac Lab task (gym registry id).")
parser.add_argument(
    "--policy_config",
    type=str,
    default="pi05_libero",
    help="OpenPI training config name (e.g. pi05_libero, pi05_libero_am_isaac).",
)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="gs://openpi-assets/checkpoints/pi05_libero",
    help="Checkpoint directory (local path or gs://...).",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--num_rollouts", type=int, default=20, help="Number of rollouts to evaluate.")
parser.add_argument("--max_timesteps", type=int, default=1500, help="Max timesteps per rollout.")
parser.add_argument("--prompt", type=str, default="press the button", help="Language instruction prompt.")
parser.add_argument(
    "--adapter",
    choices=["auto", "libero", "am_isaac"],
    default="auto",
    help="Which OpenPI input schema to use. 'auto' picks from policy_config.",
)
parser.add_argument(
    "--action_horizon",
    type=int,
    default=None,
    help="How many steps to execute per policy query. Defaults to model's action_horizon.",
)
parser.add_argument("--pos_scale", type=float, default=0.05, help="Scale position deltas for LIBERO actions.")
parser.add_argument("--rot_scale", type=float, default=0.5, help="Scale axis-angle deltas for LIBERO actions.")

# JAX/OpenPI runtime knobs.
# Isaac Sim typically owns the CUDA context; to avoid allocator/preallocation conflicts,
# we default to interop-safe settings and allow overriding.
parser.add_argument(
    "--jax_platform",
    choices=["auto", "cuda", "cpu"],
    default="auto",
    help="JAX platform. 'auto' keeps default behavior; set 'cpu' to avoid CUDA context issues.",
)
parser.add_argument(
    "--jax_preallocate",
    action="store_true",
    help="Enable JAX GPU preallocation (NOT recommended with Isaac Sim).",
)
parser.add_argument(
    "--jax_allocator",
    choices=["platform", "bfc", "default"],
    default="platform",
    help="JAX allocator. 'platform' is most interoperable with other CUDA users.",
)
parser.add_argument(
    "--jax_mem_fraction",
    type=float,
    default=None,
    help="Optional JAX GPU memory fraction (e.g. 0.2). Only applies when using CUDA.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True


def _configure_jax_runtime(args: argparse.Namespace) -> None:
    """Configure env vars that JAX/XLA reads at import time.

    This is intentionally done before importing OpenPI/JAX to reduce the chance of
    CUDA allocator/context conflicts with Isaac Sim.
    """

    # Platform selection.
    if getattr(args, "jax_platform", "auto") in ("cpu", "cuda"):
        os.environ.setdefault("JAX_PLATFORMS", str(args.jax_platform))

    # Preallocation and allocator settings matter most for CUDA.
    # Disable preallocation by default to avoid grabbing most VRAM.
    if getattr(args, "jax_preallocate", False):
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    else:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    allocator = getattr(args, "jax_allocator", "platform")
    if allocator == "default":
        # Don't force an allocator.
        pass
    else:
        os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", allocator)

    mem_fraction = getattr(args, "jax_mem_fraction", None)
    if mem_fraction is not None:
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(float(mem_fraction)))


_configure_jax_runtime(args_cli)

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import time
from dataclasses import dataclass
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

import isaaclab.utils.math as math_utils
from isaaclab_tasks.utils import parse_env_cfg

import am_isaac.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401


@dataclass(frozen=True)
class PlannedChunk:
    t0: int
    ref_pos_w: torch.Tensor  # (3,)
    ref_quat_w: torch.Tensor  # (4,) wxyz
    actions: np.ndarray  # (H, A)


def _resize_uint8_hwc(image_hwc_u8: torch.Tensor, *, size_hw: tuple[int, int]) -> np.ndarray:
    """Resize a torch uint8 HWC image to uint8 HWC numpy."""
    if image_hwc_u8.dtype != torch.uint8:
        image_hwc_u8 = image_hwc_u8.to(torch.uint8)
    # torch interpolate expects NCHW float
    img = image_hwc_u8.permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img = F.interpolate(img, size=size_hw, mode="bilinear", align_corners=False)
    img = (img[0].permute(1, 2, 0).clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    return img.cpu().numpy()


def _get_camera_rgb_u8(env, camera_attr: str, env_idx: int = 0) -> torch.Tensor:
    """Fetch RGB uint8 HWC image tensor for a given camera attribute on the env."""
    camera = getattr(env, camera_attr, None)
    if camera is None:
        raise AttributeError(f"Environment has no camera attribute '{camera_attr}'.")
    rgb = camera.data.output.get("rgb", None)
    if rgb is None:
        raise KeyError(f"Camera '{camera_attr}' has no 'rgb' in data.output.")
    # Expected shape: (num_envs, H, W, 3), dtype uint8
    return rgb[env_idx]


def _axis_angle_from_quat_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    aa = math_utils.axis_angle_from_quat(quat_wxyz.reshape(1, 4))
    return aa.reshape(3)


def _quat_from_axis_angle(axis_angle: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """axis_angle: (3,) where norm is angle."""
    angle = torch.linalg.norm(axis_angle)
    if float(angle) < eps:
        return torch.tensor([1.0, 0.0, 0.0, 0.0], device=axis_angle.device, dtype=axis_angle.dtype)
    axis = axis_angle / angle
    return math_utils.quat_from_angle_axis(angle.reshape(1), axis.reshape(1, 3)).reshape(4)


def _euler_xyz_from_quat_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat_wxyz.reshape(1, 4))
    return torch.stack([roll[0], pitch[0], yaw[0]], dim=0)


def _build_libero_example(env, raw_obs: dict, *, prompt: str, image_size: int = 224) -> dict:
    """Build a Libero-style OpenPI input example from an IsaacLab env step."""
    obs0 = raw_obs["policy"][0]

    ee_pos = obs0["ee_pos"].detach()
    ee_quat = obs0["ee_quat"].detach()

    # Libero state: [eef_pos(3), eef_axis_angle(3), gripper_qpos(2)]
    ee_axis_angle = _axis_angle_from_quat_wxyz(ee_quat)

    # Prefer true gripper qpos (2 finger joints) if available.
    if hasattr(env, "gripper_joint_ids"):
        gripper_qpos = env.robot.data.joint_pos[0, env.gripper_joint_ids].detach().to(torch.float32)
    else:
        # Fallback: duplicate gripper_width to 2 dims.
        width = obs0["gripper_width"].detach().to(torch.float32).reshape(1)
        gripper_qpos = torch.cat([width, width], dim=0)

    state = torch.cat(
        [
            ee_pos.to(torch.float32).reshape(3),
            ee_axis_angle.to(torch.float32).reshape(3),
            gripper_qpos.to(torch.float32).reshape(2),
        ],
        dim=0,
    )
    state_np = state.cpu().numpy()

    # Images: use base_camera if present, else ee_camera; wrist uses ee_camera.
    if hasattr(env, "base_camera"):
        base_rgb = _get_camera_rgb_u8(env, "base_camera", 0)
    else:
        base_rgb = _get_camera_rgb_u8(env, "ee_camera", 0)
    wrist_rgb = _get_camera_rgb_u8(env, "ee_camera", 0)

    base_np = _resize_uint8_hwc(base_rgb, size_hw=(image_size, image_size))
    wrist_np = _resize_uint8_hwc(wrist_rgb, size_hw=(image_size, image_size))

    return {
        "observation/state": state_np,
        "observation/image": base_np,
        "observation/wrist_image": wrist_np,
        "prompt": prompt,
    }


def _select_adapter(adapter: str, policy_config_name: str) -> str:
    if adapter != "auto":
        return adapter
    name = policy_config_name.lower()
    if "am_isaac" in name:
        return "am_isaac"
    if "libero" in name:
        return "libero"
    # Default to libero because it is EE-centric and easiest to adapt.
    return "libero"


def _libero_action_to_env_action(
    chunk: PlannedChunk,
    *,
    t: int,
    pos_scale: float,
    rot_scale: float,
) -> torch.Tensor:
    """Convert a planned LIBERO action at time t into Isaac env 7D action (absolute EE target)."""
    k = t - chunk.t0
    k = int(np.clip(k, 0, chunk.actions.shape[0] - 1))

    a = torch.from_numpy(np.asarray(chunk.actions[k], dtype=np.float32)).to(chunk.ref_pos_w.device)
    if a.numel() < 7:
        raise ValueError(f"Expected >=7 action dims for LIBERO, got {a.numel()}.")

    dpos = a[0:3] * float(pos_scale)
    daxaa = a[3:6] * float(rot_scale)
    gripper = a[6]

    target_pos = chunk.ref_pos_w + dpos
    delta_quat = _quat_from_axis_angle(daxaa)
    target_quat = math_utils.quat_mul(chunk.ref_quat_w.reshape(1, 4), delta_quat.reshape(1, 4)).reshape(4)
    target_rpy = _euler_xyz_from_quat_wxyz(target_quat)

    env_action = torch.cat([target_pos.reshape(3), target_rpy.reshape(3), gripper.reshape(1)], dim=0)
    return env_action


def main() -> None:
    args = args_cli

    # Create policy
    # Import OpenPI lazily so JAX reads env vars configured above.
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    train_cfg = _config.get_config(args.policy_config)
    policy = _policy_config.create_trained_policy(
        train_cfg,
        args.checkpoint_dir,
        default_prompt=args.prompt,
    )

    # Determine horizon
    horizon = int(args.action_horizon or getattr(train_cfg.model, "action_horizon", 10))

    # Create environment
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    # Some env cfgs may expose recorders; avoid hard dependency on attribute.
    if getattr(env_cfg, "recorders", None) is not None:
        setattr(env_cfg, "recorders", {})
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env_any = cast(Any, env)

    adapter = _select_adapter(args.adapter, args.policy_config)
    print("=" * 60)
    print("OpenPI Isaac Evaluation")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Policy config: {args.policy_config}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Adapter: {adapter}")
    print(f"Num rollouts: {args.num_rollouts}")
    print(f"Num envs: {args.num_envs}")
    print(f"Max timesteps: {args.max_timesteps}")
    print(f"Action horizon: {horizon}")
    print("=" * 60)

    successes: list[bool] = []

    def first_bool(x: Any) -> bool:
        if isinstance(x, torch.Tensor):
            return bool(x.flatten()[0].item())
        if isinstance(x, (list, tuple, np.ndarray)):
            return bool(x[0])
        return bool(x)

    def first_reward(x: Any) -> float:
        if isinstance(x, torch.Tensor):
            return float(x.flatten()[0].item())
        if isinstance(x, (list, tuple, np.ndarray)):
            return float(x[0])
        return float(x)

    def build_example(raw_obs: dict) -> dict:
        if adapter == "libero":
            return _build_libero_example(env_any, raw_obs, prompt=args.prompt)
        if adapter == "am_isaac":
            # Schema consumed by openpi.policies.am_isaac_policy.AmIsaacLiberoInputs.
            obs0 = raw_obs["policy"][0]

            ee_rgb = _get_camera_rgb_u8(env_any, "ee_camera", 0)
            ee_np = _resize_uint8_hwc(ee_rgb, size_hw=(224, 224))

            if hasattr(env_any, "gripper_joint_ids"):
                gripper_qpos = (
                    env_any.robot.data.joint_pos[0, env_any.gripper_joint_ids]
                    .detach()
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )
            else:
                width = float(obs0["gripper_width"].detach().cpu().item())
                gripper_qpos = np.asarray([width, width], dtype=np.float32)

            return {
                "am_isaac/ee_pos": obs0["ee_pos"].detach().to(torch.float32).cpu().numpy(),
                "am_isaac/ee_quat": obs0["ee_quat"].detach().to(torch.float32).cpu().numpy(),
                "am_isaac/gripper_qpos": gripper_qpos,
                "am_isaac/ee_image": ee_np,
                "prompt": args.prompt,
            }
        raise ValueError(f"Unsupported adapter: {adapter}")

    with contextlib.suppress(KeyboardInterrupt):
        for rollout_idx in range(args.num_rollouts):
            if not simulation_app.is_running() or simulation_app.is_exiting():
                break

            raw_obs, _ = env.reset()
            planned: PlannedChunk | None = None
            success = False

            for t in range(args.max_timesteps):
                if not simulation_app.is_running() or simulation_app.is_exiting():
                    break

                # (Re-)plan when needed
                if planned is None or (t - planned.t0) >= horizon:
                    # Reference state for delta-to-absolute conversion
                    obs0 = raw_obs["policy"][0]
                    ref_pos = obs0["ee_pos"].detach().to(torch.float32)
                    ref_quat = obs0["ee_quat"].detach().to(torch.float32)

                    example = build_example(raw_obs)
                    out = policy.infer(example)
                    actions = np.asarray(out["actions"], dtype=np.float32)
                    planned = PlannedChunk(t0=t, ref_pos_w=ref_pos, ref_quat_w=ref_quat, actions=actions)

                # Convert action for this step
                action_1 = _libero_action_to_env_action(
                    planned, t=t, pos_scale=args.pos_scale, rot_scale=args.rot_scale
                ).reshape(1, 7)

                # Expand to all envs
                if action_1.shape[0] != env_any.num_envs:
                    actions_env = action_1.repeat(env_any.num_envs, 1)
                else:
                    actions_env = action_1

                raw_obs, reward, terminated, truncated, _info = env.step(actions_env)

                if t % 10 == 0:
                    a0 = actions_env[0].detach().cpu().numpy()
                    print(f"rollout={rollout_idx+1} step={t} action={a0} reward={first_reward(reward):.3f}")

                if first_bool(terminated):
                    success = True
                    break
                if first_bool(truncated) or torch.isnan(actions_env).any():
                    break

            successes.append(success)
            print(f"Rollout {rollout_idx+1}/{args.num_rollouts} finished. Success={success}")

            # Avoid GPU memory creep
            torch.cuda.empty_cache()
            time.sleep(0.25)

    success_rate = float(np.mean(successes)) if successes else 0.0
    print("=" * 60)
    print(f"Success rate: {success_rate*100:.2f}% ({int(success_rate*len(successes))}/{len(successes)})")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
