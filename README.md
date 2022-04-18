# Conditional DETR

## First of All

**Although GitHub declares that it stands with Ukraine, I have no political position at all.**

## Installation

### Requirements
The code is developed using Python 3.8 with PyTorch 1.8.1.

```shell
cd ConditionalSwinDETR
pip install -r requirements.txt
```



## Usage

### Data preparation

Download and extract COCO 2017 train and val images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).
We expect the directory structure to be the following:
```
path/to/coco/
├── annotations/  # annotation json files
└── images/
    ├── train2017/    # train images
    └── val2017/      # val images
```

### Training

To train conditional DETR-Swin-T on a single node with 8 gpus for 50 epochs run:
```shell
bash scripts/conddetr_swin_tiny_epoch50.sh
```
or
```shell
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env \
    main.py \
    --resume auto \
    --coco_path /path/to/coco \
    --output_dir output/conddetr_swin_tiny_epoch50
```
The training process takes around 30 hours on a single machine with 8 V100 cards.

Same as DETR training setting, we train conditional DETR with AdamW setting learning rate in the transformer to 1e-4 and 1e-5 in the backbone.
Horizontal flips, scales and crops are used for augmentation.
Images are rescaled to have min size 800 and max size 1333.
The transformer is trained with dropout of 0.1, and the whole model is trained with grad clip of 0.1.

### Evaluation
To evaluate conditional DETR-Swin-T on COCO *val* with 8 GPUs run:
```shell
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env \
    main.py \
    --batch_size 2 \
    --eval \
    --resume <checkpoint.pth> \
    --coco_path /path/to/coco \
    --output_dir output/<output_path>
```

### Inference & Visualization

```bash
bash scripts/conddetr_swin_tiny_inference.sh
```

or

```bash
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --use_env \
    inference.py \
    --img_path ./demo.jpg
```

## License

Conditional DETR is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.



## References

```bibtex
@inproceedings{meng2021-CondDETR,
  title       = {Conditional DETR for Fast Training Convergence},
  author      = {Meng, Depu and Chen, Xiaokang and Fan, Zejia and Zeng, Gang and Li, Houqiang and Yuan, Yuhui and Sun, Lei and Wang, Jingdong},
  booktitle   = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year        = {2021}
}
```

```bibtex
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```

