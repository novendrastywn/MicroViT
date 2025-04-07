
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from .microvit import MicroViT
from timm.models.registry import register_model      

#=============================MicroViT==========================================
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

def reparameterize(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'reparam'):
            reparametrized = child.reparam()
            setattr(net, child_name, reparametrized)
            reparameterize(reparametrized)
        # elif isinstance(child, torch.nn.BatchNorm2d):
            # setattr(net, child_name, torch.nn.Identity())
        else:
            reparameterize(child)
