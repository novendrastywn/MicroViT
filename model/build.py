'''
Build the SHViT model family
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.registry import register_model
from .microvit import MicroViT
from .microvitv2 import MicroViTv2

@register_model
def microvit_1(pretrained=False, **kwargs):
    model=MicroViT(
        dims   = [ 128, 256, 320],
        depths = [ 2, 5, 5],
        type   = [ 'c', 'c', 'a'],
        qk_dim  = [0, 0, 16],
        attn_sr  = [ 0, 0, 1],
        attn_ipg = [ 0, 0, 32],
        attn_cr  = [ 0, 0, 0.215],
        mlp_ratio = 2,
        patch_size = 16,
        act_layer = nn.GELU,
        final_feature=None,
        **kwargs
        )
    # reparameterize(model)
    return model

@register_model
def microvit_2(pretrained=False, **kwargs):
    model=MicroViT(
        dims=[128, 320, 448],
        depths=[ 2, 7, 5],
        type=[ 'c', 'c', 'a'],
        qk_dim = [0, 0, 16],
        attn_sr=[ 0, 0, 1],
        attn_ipg=[ 0, 0, 32],
        attn_cr=[ 0, 0, 0.215],
        mlp_ratio=2,
        act_layer=nn.GELU,
        patch_size=16,
        final_feature=None,
        **kwargs
        )
    # reparameterize(model)
    return model

@register_model
def microvit_3(pretrained=False, **kwargs):
    model=MicroViT(
        dims=[ 192, 384, 512],
        depths=[ 3, 7, 6],
        type=[ 'c', 'c', 'a'],
        qk_dim = [0, 0, 16],
        attn_sr=[ 0, 0, 1],
        attn_ipg=[ 0, 0, 32],
        attn_cr=[ 0, 0, 0.215],
        mlp_ratio=2,
        patch_size=16,
        final_feature=None,
        act_layer=nn.GELU,
        **kwargs
        )
    # reparameterize(model)
    return model

@register_model
def microvitv2_1(pretrained=False, **kwargs):
    model=MicroViTv2(
        dims   = [ 128, 224, 320],
        depths = [ 2, 6, 5],
        type   = [ 'c', 'c', 'sdta'],
        attn_cr = [ 0, 0, 0.25],
        patch_size = 16,
        act_layer = nn.SiLU,
        final_feature=None,
        **kwargs
        )
    # reparameterize(model)
    return model

@register_model
def microvitv2_2(pretrained=False, **kwargs):
    model=MicroViTv2(
        dims   = [ 128, 308, 448],
        depths = [ 2, 7, 5],
        type   = [ 'c', 'c', 'sdta'],
        attn_cr = [ 0, 0, 0.25],
        patch_size = 16,
        act_layer = nn.SiLU,
        final_feature=None,
        **kwargs
        )
    # reparameterize(model)
    return model

@register_model
def microvitv2_2_mdta(pretrained=False, **kwargs):
    model=MicroViTv2(
        dims   = [ 128, 308, 448],
        depths = [ 2, 7, 5],
        type   = [ 'c', 'c', 'mdta'],
        attn_cr = [ 0, 0, 0.25],
        patch_size = 16,
        act_layer = nn.SiLU,
        final_feature=None,
        **kwargs
        )
    # reparameterize(model)
    return model

@register_model
def microvitv2_3(pretrained=False, **kwargs):
    model=MicroViTv2(
        dims   = [ 192, 384, 448],
        depths = [ 3, 7, 6],
        type   = [ 'c', 'c', 'sdta'],
        attn_cr = [ 0, 0, 0.25],
        patch_size = 16,
        act_layer = nn.SiLU,
        final_feature=None,
        **kwargs
        )
    # reparameterize(model)
    return model

def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            fused = child.fuse()
            setattr(net, child_name, fused)
            replace_batchnorm(fused)
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)

def reparameterize(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'reparam'):
            reparametrized = child.reparam()
            setattr(net, child_name, reparametrized)
            reparameterize(reparametrized)
        elif hasattr(child, 'fuse'):
            reparametrized = child.fuse()
            setattr(net, child_name, reparametrized)
            reparameterize(reparametrized)
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            reparameterize(child)
    
    return net