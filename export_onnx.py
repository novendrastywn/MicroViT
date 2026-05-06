import timm.utils
import torch
import onnxruntime as ort
import time
import timm
from timm import create_model
from model.build import *
from model.build import replace_batchnorm, reparameterize
from model.iformer import *
# from model.microvitv2_abl import GConvFFN
# from model import build
# import models.emo_model.emo
import utils
from fvcore.nn import FlopCountAnalysis, parameter_count
torch.autograd.set_grad_enabled(False)


T0 = 5
T1 = 10


def export_onnx(name, model, device, batch_size, inc, resolution=224):
    model.to('cpu')
    model = model.eval()
    print("Convert To ONNX...")
    inputs = torch.randn(batch_size, inc, resolution, resolution, device='cpu')
    torch.onnx.export(model, inputs, f"./onnx/{name}_{batch_size}.onnx", verbose = False, opset_version=16)
    inputs = torch.randn(1, inc, resolution, resolution, device='cpu')
    torch.onnx.export(model, inputs, f"./onnx/{name}_{1}.onnx", verbose = False, opset_version=16)

    print("Finish...")

device = "cuda:0"

from argparse import ArgumentParser
import torchvision

parser = ArgumentParser()

parser.add_argument('--model', default='fused15_c512_r4', type=str) #repinc_m1
parser.add_argument('--resolution', default=224, type=int)
parser.add_argument('--inchannel', default=3, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--checkpoint', default=None)

if __name__ == "__main__":
    args = parser.parse_args()
    model_name = args.model
    batch_size = args.batch_size
    resolution = args.resolution
    inc = args.inchannel
    torch.cuda.empty_cache()
    if args.model == 'shufflenet_v2_x1_0':
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
    elif args.model == 'shufflenet_v2_x1_5':
        model = torchvision.models.shufflenet_v2_x1_5(pretrained=True)
    elif args.model == 'shufflenet_v2_x2_0':
        model = torchvision.models.shufflenet_v2_x2_0(pretrained=True)
        # model = torchvision.models.mobilenet
    else:
        model = create_model(model_name, num_classes=1000) #inference_mode=True,
    
    # model = GConvFFN(inc, inc, f=True)

    if args.checkpoint is not None:
        print(f"Load checkpoint {args.checkpoint}")
        chkpt = torch.load(args.checkpoint, weights_only=False, map_location='cpu')
        model.load_state_dict(chkpt["model"])

    model=timm.utils.reparameterize_model(model)
    # model = reparameterize(model.eval())
    # model = reparameterize(model)
    print(model)
    inputs = torch.randn(1, inc, resolution, resolution, device='cpu')

    # torch.onnx.export(model, inputs, './onnx/'+args.model+".onnx")
    from ptflops import get_model_complexity_info
    macs, n_parameters = get_model_complexity_info(
                        model.eval(), (inc, resolution, resolution), as_strings=False,
                        print_per_layer_stat=False, verbose=False)
    gmacs = macs / (1000**3)
    print(f"{args.model}, params: {(n_parameters/(1000**2)):.2f} M, macs: {gmacs:.3f} G")
    # inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
    export_onnx(model_name, model, device='cpu', 
                batch_size=args.batch_size, inc=inc, resolution=resolution)
