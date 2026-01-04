cd ../

uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_am_isaac \
    --policy.dir=/home/ubuntu/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch