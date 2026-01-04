from openpi.policies import libero_policy, policy_config
from openpi.training import config as _config
from openpi.shared.download import DEFAULT_CACHE_DIR
import os

config = _config.get_config("pi05_libero")
checkpoint_dir = os.path.join(os.path.expanduser(DEFAULT_CACHE_DIR), "openpi-assets/checkpoints/pi05_libero_pytorch")
print(f"Using checkpoint dir: {checkpoint_dir}")

# Create a trained policy (automatically detects PyTorch format)
policy = policy_config.create_trained_policy(config, checkpoint_dir)

example = libero_policy.make_libero_example()

# Run inference (same API as JAX)
action_chunk = policy.infer(example)["actions"]

# Delete the policy to free up memory.
del policy

print("Actions shape:", action_chunk.shape)