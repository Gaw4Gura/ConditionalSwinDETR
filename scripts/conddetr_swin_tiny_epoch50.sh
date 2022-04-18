python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env \
    main.py \
    --coco_path /path/to/coco \
    --output_dir output/$script_name