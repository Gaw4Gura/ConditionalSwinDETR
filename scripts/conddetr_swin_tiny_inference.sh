python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --use_env \
    inference.py \
    --img_path ./demo.jpg