import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import am_bench_policy
from openpi.training import config as _config


@pytest.mark.parametrize(
    ("config_name", "model_type", "action_representation"),
    [
        (
            "pi0_am_bench_multitask_openpi_original_20hz_h50_ee_local_relative",
            _model.ModelType.PI0,
            "ee_local_relative",
        ),
        (
            "pi0_am_bench_multitask_base_joint_openpi_original_20hz_h50_base_joint_relative",
            _model.ModelType.PI0,
            "base_joint_relative",
        ),
        (
            "pi05_am_bench_multitask_openpi_original_20hz_h50_ee_local_relative",
            _model.ModelType.PI05,
            "ee_local_relative",
        ),
        (
            "pi05_am_bench_multitask_base_joint_openpi_original_20hz_h50_base_joint_relative",
            _model.ModelType.PI05,
            "base_joint_relative",
        ),
    ],
)
def test_supported_configs(config_name, model_type, action_representation):
    config = _config.get_config(config_name)

    assert config.model.model_type is model_type
    assert config.model.action_horizon == 50
    assert config.policy_metadata == {"action_representation": action_representation}
    assert config.pytorch_weight_path is None


@pytest.mark.parametrize(
    "config_name",
    [
        "pi0_am_bench_multitask_openpi_original_20hz_h16_ee_local_relative",
        "pi0_am_bench_multitask_base_joint_openpi_original_20hz_h16_base_joint_relative",
        "pi0_am_bench_press_button_base_joint_pid_openpi_original_20hz_h16_base_joint_relative",
        "pi0_am_bench_push_slider_base_joint_pid_openpi_original_20hz_h16_base_joint_relative",
        "pi0_am_bench_press_button_push_slider_base_joint_pid_openpi_original_20hz_h16_base_joint_relative",
        "pi0_am_bench_press_button_base_joint_l1_openpi_original_20hz_h16_base_joint_relative",
        "pi0_am_bench_push_slider_base_joint_l1_openpi_original_20hz_h16_base_joint_relative",
        "pi0_am_bench_press_button_push_slider_base_joint_l1_openpi_original_20hz_h16_base_joint_relative",
        "pi05_am_bench_press_button",
        "pi05_am_bench_multitask_openpi_original_20hz_h16_ee_local_relative",
        "pi05_am_bench_multitask_base_joint_openpi_original_20hz_h16_base_joint_relative",
    ],
)
def test_experimental_configs_are_not_registered(config_name):
    with pytest.raises(ValueError, match="not found"):
        _config.get_config(config_name)


def test_ee_local_relative_adapter_round_trip():
    cache = am_bench_policy.EELocalRelativeCache()
    input_transform = am_bench_policy.AmBenchInputs(
        model_type=_model.ModelType.PI05,
        action_representation="ee_local_relative",
        ee_local_relative_cache=cache,
    )
    output_transform = am_bench_policy.AmBenchOutputs(
        action_representation="ee_local_relative",
        ee_local_relative_cache=cache,
    )
    actions = np.array(
        [
            [0.5, -0.15, 1.2, 0.9238795, 0.0, 0.0, 0.3826834, 0.02],
            [0.4, -0.25, 1.3, 0.7071068, 0.0, 0.0, 0.7071068, 0.04],
        ],
        dtype=np.float32,
    )

    transformed = input_transform(
        {
            "am_bench/ee_pos": np.array([0.5, -0.25, 1.2], dtype=np.float32),
            "am_bench/ee_quat": np.array([0.7071068, 0.0, 0.0, 0.7071068], dtype=np.float32),
            "am_bench/gripper_width": np.array([0.03], dtype=np.float32),
            "am_bench/ee_image": np.full((3, 4, 5), 0.5, dtype=np.float32),
            "prompt": b"press the button",
            "actions": actions,
        }
    )

    np.testing.assert_allclose(transformed["state"], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.03])
    assert transformed["image"]["left_wrist_0_rgb"].shape == (4, 5, 3)
    assert not transformed["image_mask"]["base_0_rgb"]
    assert transformed["prompt"] == "press the button"
    np.testing.assert_allclose(output_transform(transformed)["actions"], actions, atol=1e-6)


def test_base_joint_relative_adapter_round_trip():
    cache = am_bench_policy.BaseJointRelativeCache()
    input_transform = am_bench_policy.AmBenchInputs(
        model_type=_model.ModelType.PI0,
        action_representation="base_joint_relative",
        base_joint_relative_cache=cache,
    )
    output_transform = am_bench_policy.AmBenchOutputs(
        action_representation="base_joint_relative",
        base_joint_relative_cache=cache,
    )
    actions = np.array(
        [
            [0.2, -0.1, 0.05, 0.9238795, 0.0, 0.0, 0.3826834, 0.2, -0.1, 0.4, -0.2, 0.03],
            [0.1, 0.1, 0.0, 0.7071068, 0.0, 0.0, 0.7071068, 0.3, -0.2, 0.5, -0.3, 0.01],
        ],
        dtype=np.float32,
    )

    transformed = input_transform(
        {
            "am_bench/base_pos": np.array([0.1, -0.2, 0.0], dtype=np.float32),
            "am_bench/base_quat": np.array([0.7071068, 0.0, 0.0, 0.7071068], dtype=np.float32),
            "am_bench/arm_joint_pos": np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32),
            "am_bench/gripper_width": np.array([0.02], dtype=np.float32),
            "am_bench/ee_image": np.zeros((4, 5, 3), dtype=np.uint8),
            "actions": actions,
        }
    )

    np.testing.assert_allclose(
        transformed["state"],
        [0.1, -0.2, 0.0, 0.7071068, 0.0, 0.0, 0.7071068, 0.1, -0.2, 0.3, -0.4, 0.02],
    )
    np.testing.assert_allclose(output_transform(transformed)["actions"], actions, atol=1e-6)
