# MicroViT: Vision Transformers with Low-Complexity Attention for Edge Devices

This repository hosts the official code for the **MicroViT** family of lightweight Vision Transformers designed for energy-efficient edge deployment.

| Version | Paper | Venue | Status |
|:---|:---|:---:|:---:|
| **MicroViTv2** | [MicroViTv2: Beyond the FLOPs for Edge Energy-Friendly Vision Transformers](#) | ICIP 2026 | ✅ Accepted |
| **MicroViTv1** | [MicroViT: A Vision Transformer with Low Complexity Self Attention for Edge Device](https://arxiv.org/abs/2502.05800) | ISCAS 2025 | ✅ Published |

*Novendra Setyawan, Chi-Chia Sun, Mao-Hsiu Hsu, Wen-Kai Kuo, Jun-Wei Hsieh*

---

## What's new in MicroViTv2

MicroViTv2 builds on the original MicroViT (now referred to as **MicroViTv1**) with three hardware-aware design changes targeting *real-device* efficiency rather than FLOPs reduction:

- **RepEmbed** — Re-parameterized patch embedding with parallel 3×3, 1×1, and identity branches at training time, fused into a single convolution at inference.
- **RepDW** — Re-parameterized depth-wise convolution mixer for the first two pyramid stages, extending the same fuse-at-inference principle to the spatial mixer.
- **SDTA** — Single Depth-Wise Transposed Attention. Evolves ESHA from MicroViTv1 by adding transposed attention with explicit local–global fusion, capturing long-range dependencies at channel-based complexity (linear in spatial size).

On Jetson AGX Orin, MicroViTv2 surpasses MicroViTv1, MobileViTv2, EdgeNeXt, EfficientViT, and SHViT in the joint accuracy / throughput / energy trade-off — despite slightly higher theoretical FLOPs, validating that **FLOPs alone do not predict edge efficiency**.

<details>
  <summary><font size="+1">MicroViTv2 Abstract</font></summary>

The Vision Transformer (ViT) achieves remarkable accuracy across visual tasks but remains computationally expensive for edge deployment. This paper presents MicroViTv2, a lightweight Vision Transformer optimized for real-device efficiency. Built upon the original MicroViT, the proposed model is designed based on a re-parameterized design — specifically Reparameterize Patch Embedding (RepEmbed) and Reparameterize Depth-Wise convolution mixer (RepDW) — for faster inference, and introduces the Single Depth-Wise Transposed Attention (SDTA) to capture long-range dependencies with minimal redundancy. Despite slightly higher FLOPs, MicroViTv2 improves accuracy by up to 0.5% compared to its predecessor and surpasses MobileViTv2, EdgeNeXt, and EfficientViT while maintaining fast inference and high energy efficiency on Jetson AGX Orin. Experiments on ImageNet-1K and COCO demonstrate that hardware-aware design and structural re-parameterization are key to achieving high accuracy and low energy consumption, validating the need to evaluate efficiency beyond FLOPs.

</details>

<details>
  <summary><font size="+1">MicroViTv1 Abstract</font></summary>

The Vision Transformer (ViT) has demonstrated state-of-the-art performance in various computer vision tasks, but its high computational demands make it impractical for edge devices with limited resources. This paper presents MicroViT, a lightweight Vision Transformer architecture optimized for edge devices by significantly reducing computational complexity while maintaining high accuracy. The core of MicroViT is the Efficient Single Head Attention (ESHA) mechanism, which utilizes group convolution to reduce feature redundancy and processes only a fraction of the channels, thus lowering the burden of the self-attention mechanism. MicroViT is designed using a multi-stage MetaFormer architecture, stacking multiple MicroViT encoders to enhance efficiency and performance. Comprehensive experiments on the ImageNet-1K and COCO datasets demonstrate that MicroViT achieves competitive accuracy while significantly improving 3.6× faster inference speed and reducing energy consumption with 40% higher efficiency than the MobileViT series, making it suitable for deployment in resource-constrained environments such as mobile and edge devices.

</details>

---

## Pre-trained Models

### MicroViTv2 (ICIP 2026) — measured on NVIDIA Jetson AGX Orin

| Model           | Res. | Param (M) | FLOPs (M) | Top-1 (%) | Latency (ms) | Throughput (Img/s) | Energy (mJ/Img) | η (%/(mJ/Img)) | Weights |
|:---             |:---: |:---:      |:---:      |:---:      |:---:         |:---:               |:---:            |:---:           |:---:    |
| MicroViTv2-S1   | 224  | 6.7       | 250       | 72.7      | 4.84         | 2367.6             | 10.9            | **6.67**       | _coming soon_ |
| MicroViTv2-S2   | 224  | 12.7      | 407       | 75.1      | 5.25         | 1883.3             | 14.9            | **5.04**       | _coming soon_ |
| MicroViTv2-S3   | 224  | 17.0      | 676       | 77.4      | 5.90         | 1335.5             | 22.5            | **3.44**       | _coming soon_ |

### MicroViTv1 (ISCAS 2025)

| Model       | Res. | Param (M) | FLOPs (M) | GPU (Img/s) | CPU (Img/s) | Top-1 (%) | Weights |
|:---         |:---: |:---:      |:---:      |:---:        |:---:        |:---:      |:---:    |
| MicroViT-S1 | 224  | 6.4       | 231       | 17466       | 552         | 72.6      | [model](https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s1.pth) |
| MicroViT-S2 | 224  | 10.0      | 345       | 14154       | 435         | 74.6      | [model](https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s2.pth) |
| MicroViT-S3 | 224  | 16.7      | 580       | 9288        | 232         | 77.1      | [model](https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s3.pth) |

> **η** (energy efficiency) = Top-1 Accuracy (%) / Energy per Image (mJ). Higher is better.

---

## Citation

If our work or code helps your research, please cite the relevant paper(s):

**MicroViTv2 (ICIP 2026):**
```bibtex
@inproceedings{setyawan2026microvitv2,
  title     = {MicroViTv2: Beyond the FLOPs for Edge Energy-Friendly Vision Transformers},
  author    = {Setyawan, Novendra and Sun, Chi-Chia and Hsu, Mao-Hsiu and Kuo, Wen-Kai and Hsieh, Jun-Wei},
  booktitle = {IEEE int. conf. on image process. (ICIP)},
  year      = {2026}
}
```

**MicroViTv1 (ISCAS 2025):**
```bibtex
@inproceedings{setyawan2025microvit,
  title     = {MicroViT: A Vision Transformer with Low Complexity Self Attention for Edge Device},
  author    = {Setyawan, Novendra and Sun, Chi-Chia and Hsu, Mao-Hsiu and Kuo, Wen-Kai and Hsieh, Jun-Wei},
  booktitle = {IEEE int. symp. on circ. and syst. (ISCAS)},
  pages     = {1--5},
  year      = {2025}
}
```

---

## Acknowledgements

We sincerely appreciate [Swin Transformer](https://github.com/microsoft/swin-transformer), [LeViT](https://github.com/facebookresearch/LeViT), [pytorch-image-models](https://github.com/rwightman/pytorch-image-models), [EfficientViT](https://github.com/microsoft/Cream/tree/main/EfficientViT), [SHViT](https://github.com/ysj9909/SHViT), [RepVGG](https://github.com/DingXiaoH/RepVGG), [Restormer](https://github.com/swz30/Restormer), and [PyTorch](https://github.com/pytorch/pytorch) for their wonderful implementations.
