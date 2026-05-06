"""
Testing the speed of different models
"""
import os
import torch
import torchvision
import time
import timm
import onnx
import onnxruntime as ort
import numpy as np
from model.build import *
from model.build import reparameterize
import torchvision
import utils
from ptflops import get_model_complexity_info
torch.autograd.set_grad_enabled(False)


T0 = 10
T1 = 60


def compute_throughput_cpu(name, model, device, batch_size, resolution=224):
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
    # warmup
    start = time.time()
    while time.time() - start < T0:
        model(inputs)

    timing = []
    while sum(timing) < T1:
        start = time.time()
        model(inputs)
        timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    print(f"{name}, {device}, {(batch_size / timing.mean().item()):.3f}, img/s @ batch size, {batch_size}")


def compute_throughput_cpu_onnx(name, model, device, batch_size, resolution=224):
    inputs = torch.randn(batch_size, 3, resolution, resolution, device='cpu')
    torch.onnx.export(model, inputs, f"{name}.onnx", verbose = False, opset_version=11)
    #torch.onnx.export(model, inputs, f"{name}.onnx", verbose = False)
    ort_session = ort.InferenceSession(f"{name}.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: inputs.numpy()}

    start = time.time()
    while time.time() - start < T0:
        _ = ort_session.run(None, ort_inputs)
    
    timing = []
    while sum(timing) < T1:
        start = time.time()
        _ = ort_session.run(None, ort_inputs)
        timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    print(f"{name}, {device}, {(batch_size / timing.mean().item()):.3f}, img/s @ batch size, {batch_size}")
    print(f"{name}, {device}, {1000/(batch_size / timing.mean().item()):.3f}, ms latency @ batch size, {batch_size}")

def compute_throughput_cuda(name, model, device, batch_size, resolution=224):
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.time()
    with torch.amp.autocast('cuda'):
        while time.time() - start < T0:
            model(inputs)
    timing = []
    if device == 'cuda:0':
        torch.cuda.synchronize()
    with torch.amp.autocast('cuda'):
        while sum(timing) < T1:
            start = time.time()
            model(inputs)
            torch.cuda.synchronize()
            timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    print(f"{name}, {device}, {(batch_size / timing.mean().item()):.3f} img/s @ batch size, {batch_size}")


for device in ['cuda:0']: # 'cpu', 'cpu_onnx', 'cuda:0'
    if 'cuda' in device and not torch.cuda.is_available():
        print("no cuda")
        continue

    if device == 'cpu':
        os.system('echo -n "nb processors "; '
                  'cat /proc/cpuinfo | grep ^processor | wc -l; '
                  'cat /proc/cpuinfo | grep ^"model name" | tail -1')
        print('Using 1 cpu thread')
        torch.set_num_threads(1)
        compute_throughput = compute_throughput_cpu
    elif device == 'cpu_onnx':
        os.system('echo -n "nb processors "; '
                  'cat /proc/cpuinfo | grep ^processor | wc -l; '
                  'cat /proc/cpuinfo | grep ^"model name" | tail -1')
        print('Using 1 cpu thread')
        torch.set_num_threads(1)
        compute_throughput = compute_throughput_cpu_onnx
    else:
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
        compute_throughput = compute_throughput_cuda

    for n, batch_size0, resolution in [
        # ('shvit_s2', 256, 224),
        ('microvitv2_2', 256, 224),
        # ('shvit_s2', 256, 224),
        # ('shvit_s4', 256, 256),
    ]:

        if device == 'cpu' or device == 'cpu_onnx':
            batch_size = 16
        else:
            batch_size = batch_size0
            torch.cuda.empty_cache()
        inputs = torch.randn(batch_size, 3, resolution,
                             resolution, device=device)
        model = eval(n)(num_classes=1000)
        model = reparameterize(model) 
        # utils.replace_batchnorm(model)
        print(model)
        macs, params = get_model_complexity_info(
                        model, (3, resolution, resolution), as_strings=False,
                        print_per_layer_stat=False, verbose=False)
        gmacs = macs / (1000**3)
        print(f"{n}, params: {(params/(1000**2)):.2f} M, macs: {gmacs:.3f} G")

        model.to(device)
        model.eval()
        model = torch.jit.trace(model, inputs)
        compute_throughput(n, model, device, batch_size, resolution=resolution)
