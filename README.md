# MicroViT: A Vision Transformer with Low Complexity Self Attention for Edge Device

This is the official repository of 

[**MicroViT: A Vision Transformer with Low Complexity Self Attention for Edge Device**](https://arxiv.org/abs/2502.05800)
*Novendra Setyawan, Chi-Chia Sun, Mao-Hsiu Hsu, Wen-Kai Kuo, Jun-Wei Hsieh.* ISCAS 2025

<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
The Vision Transformer (ViT) has demonstrated state-of-the-art performance in various computer vision tasks, but its high computational demands make it impractical for edge devices with limited resources. This paper presents MicroViT, a lightweight Vision Transformer architecture optimized for edge devices by significantly reducing computational complexity while maintaining high accuracy. The core of MicroViT is the Efficient Single Head Attention (ESHA) mechanism, which utilizes group convolution to reduce feature redundancy and processes only a fraction of the channels, thus lowering the burden of the self-attention mechanism. MicroViT is designed using a multi-stage MetaFormer architecture, stacking multiple MicroViT encoders to enhance efficiency and performance. Comprehensive experiments on the ImageNet-1K and COCO datasets demonstrate that MicroViT achieves competitive accuracy while significantly improving 3.6 faster inference speed and reducing energy consumption with 40% higher efficiency than the MobileViT series, making it suitable for deployment in resource-constrained environments such as mobile and edge devices.
</details>


## Pre-trained Models
| Model | Resolution | Param | FLOPs | GPU | CPU | Top-1|
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|
| MicroViT-S1 | 224 | 6.4  | 0.231 | 17466 | 552 | 72.6 |[model](https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s1.pth) |
| MicroViT-S2 | 224 | 10.0 | 0.345 | 14154 | 435 | 74.6 |[model](https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s2.pth) |
| MicroViT-S3 | 224 | 16.7 | 0.580 | 9288  | 232 | 77.1 |[model](https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s3.pth) |

## Training
### Image Classification

#### Setup
```bash
conda create -n microvit python=3.9
conda activate microvit
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

#### Dataset Preparation

Download the [ImageNet-1K](http://image-net.org/) dataset and structure the data as follows:
```
/path/to/imagenet-1k/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  validation/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```

To train SHViT models, follow the respective command below:
<details>
<summary>
SHViT-S1
</summary>

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 --use_env main.py --model shvit_s1 --data-path $PATH_TO_IMAGENET --dist-eval --weight-decay 0.025
```
</details>

<details>
<summary>
SHViT-S2
</summary>

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 --use_env main.py --model shvit_s2 --data-path $PATH_TO_IMAGENET --dist-eval --weight-decay 0.032
```
</details>

<details>
<summary>
SHViT-S3
</summary>

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 --use_env main.py --model shvit_s3 --data-path $PATH_TO_IMAGENET --dist-eval --weight-decay 0.035
```
</details>

<details>
<summary>
SHViT-S4
</summary>

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 --use_env main.py --model shvit_s4 --data-path $PATH_TO_IMAGENET --dist-eval --weight-decay 0.03 --input-size 256
```
</details>


## Evaluation
Run the following command to evaluate a pre-trained SHViT-S4 on ImageNet-1K validation set with a single GPU:
```bash
python main.py --eval --model shvit_s4 --resume ./shvit_s4.pth --data-path $PATH_TO_IMAGENET --input-size 256
```


## Latency Measurement
Run the following command to compare the throughputs on GPU/CPU:

```
python speed_test.py
```

The mobile latency reported in SHViT for iPhone 12 uses the deployment tool from [XCode 14](https://developer.apple.com/videos/play/wwdc2022/10027/).

export the model to Core ML model

```
python export_model.py --variant shvit_s4 --output-dir /path/to/save/exported_model \
--checkpoint /path/to/pretrained_checkpoints/shvit_s4.pth
```

## Citation
If our work or code help your work, please cite our paper:
```
@inproceedings{yun2024shvit,
  author={Yun, Seokju and Ro, Youngmin},
  title={SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={5756--5767},
  year={2024}
}
```

## Acknowledgements
We sincerely appreciate [Swin Transformer](https://github.com/microsoft/swin-transformer), [LeViT](https://github.com/facebookresearch/LeViT), [pytorch-image-models](https://github.com/rwightman/pytorch-image-models), [EfficientViT](https://github.com/microsoft/Cream/tree/main/EfficientViT) and [PyTorch](https://github.com/pytorch/pytorch) for their wonderful implementations.
