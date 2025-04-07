import torch

from timm import create_model
import timm
# import models.efficientnext.EfficientNeXt
import models.build
# import models.emo_model.emo

import utils

import torch
import torchvision
from argparse import ArgumentParser
from fvcore.nn import FlopCountAnalysis, parameter_count

parser = ArgumentParser()

parser.add_argument('--model', default='efficientnext_a', type=str)
parser.add_argument('--resolution', default=224, type=int)
parser.add_argument('--ckpt', default=None, type=str)

if __name__ == "__main__":
    # Load a pre-trained version of MobileNetV2
    args = parser.parse_args()
    model = create_model(args.model) #inference_mode=True, distillation=True
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt)['model'])
    # utils.replace_batchnorm(model)
    model=timm.utils.reparameterize_model(model) 
    model.eval()

    # Trace the model with random data.
    resolution = args.resolution
    example_input = torch.rand(1, 3, resolution, resolution) 
    traced_model = torch.jit.trace(model, torch.Tensor(example_input))
    out = traced_model(example_input)
    # inputs = torch.randn(1, 3, resolution, resolution, device='cpu')
    # torch.onnx.export(model, inputs, './onnx/'+args.model+".onnx")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fvcore_param = parameter_count(model)
    flops = FlopCountAnalysis(model, example_input)
    print(f'{args.model}, Param:{(fvcore_param[""]/1e6):.1f}  flops: {(flops.total() / 1e9):0.2f}')
    print(f'Export CoreML model')

    import coremltools as ct

    # Using image_input in the inputs parameter:
    # Convert to Core ML neural network using the Unified Conversion API.
    model = ct.convert(
        traced_model,
        convert_to="neuralnetwork",
        inputs=[ct.ImageType(shape=example_input.shape)]
    )

    # Save the converted model.
    model.save(f"./coreml/{args.model}_{resolution}.mlmodel")