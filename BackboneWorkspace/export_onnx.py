import timm.utils
import torch
import onnxruntime as ort
import time
import timm
from timm import create_model
import models.build
import models.emo_model.emo
import utils
from fvcore.nn import FlopCountAnalysis, parameter_count
torch.autograd.set_grad_enabled(False)

T0 = 5
T1 = 10


def export_onnx(name, model, device, batch_size, resolution=224):
    model.to('cpu')
    model = model.eval()
    print("Convert To ONNX...")
    inputs = torch.randn(batch_size, 3, resolution, resolution, device='cpu')
    torch.onnx.export(model, inputs, f"./onnx/{name}_{batch_size}.onnx", verbose = False, opset_version=16)
    inputs = torch.randn(1, 3, resolution, resolution, device='cpu')
    torch.onnx.export(model, inputs, f"./onnx/{name}_{1}.onnx", verbose = False, opset_version=16)

    print("Finish...")

device = "cuda:0"

from argparse import ArgumentParser
import torchvision

parser = ArgumentParser()

parser.add_argument('--model', default='repinc_m2_3', type=str) #repinc_m1
parser.add_argument('--resolution', default=224, type=int)
parser.add_argument('--batch-size', default=64, type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    model_name = args.model
    batch_size = args.batch_size
    resolution = args.resolution
    torch.cuda.empty_cache()
    if args.model == 'shufflenet_v2_x1_0':
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
    elif args.model == 'shufflenet_v2_x1_5':
        model = torchvision.models.shufflenet_v2_x1_5(pretrained=True)
    elif args.model == 'shufflenet_v2_x2_0':
        model = torchvision.models.shufflenet_v2_x2_0(pretrained=True)
        # model = torchvision.models.mobilenet
    else:
        model = create_model(model_name, num_classes=1000).eval() #inference_mode=True,

    # model=timm.utils.reparameterize_model(model) 
    # model = fuse_conv_bn(model)
    # print(model)
    inputs = torch.randn(1, 3, resolution, resolution, device='cpu')
    # torch.onnx.export(model, inputs, './onnx/'+args.model+".onnx")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fvcore_param = parameter_count(model)
    flops = FlopCountAnalysis(model, inputs)
    print(f'Param:{fvcore_param[""]/1e6}  flops: {flops.total() / 1e9}')
    # inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
    export_onnx(model_name, model, device='cpu', 
                batch_size=args.batch_size, resolution=resolution)
