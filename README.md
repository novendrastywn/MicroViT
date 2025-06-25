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
| Model | Resolution | Param | FLOPs | GPU | CPU | Top-1| Link |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:| :---:|
| MicroViT-S1 | 224 | 6.4  | 0.231 | 17466 | 552 | 72.6 |[model](https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s1.pth) |
| MicroViT-S2 | 224 | 10.0 | 0.345 | 14154 | 435 | 74.6 |[model](https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s2.pth) |
| MicroViT-S3 | 224 | 16.7 | 0.580 | 9288  | 232 | 77.1 |[model](https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s3.pth) |




## Citation
If our work or code help your work, please cite our paper:
```
@article{setyawan2025microvit,
  title={MicroViT: A Vision Transformer with Low Complexity Self Attention for Edge Device},
  author={Setyawan, Novendra and Sun, Chi-Chia and Hsu, Mao-Hsiu and Kuo, Wen-Kai and Hsieh, Jun-Wei},
  journal={arXiv preprint arXiv:2502.05800},
  year={2025}
}
```

## Acknowledgements
We sincerely appreciate [Swin Transformer](https://github.com/microsoft/swin-transformer), [LeViT](https://github.com/facebookresearch/LeViT), [pytorch-image-models](https://github.com/rwightman/pytorch-image-models), [EfficientViT](https://github.com/microsoft/Cream/tree/main/EfficientViT) and [PyTorch](https://github.com/pytorch/pytorch) for their wonderful implementations.
