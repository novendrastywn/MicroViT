
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from .revit import REViT
from timm.models.registry import register_model

@register_model
def revit_sa(pretrained=False, num_classes = 1000, distillation=False, **kwargs):
    model=REViT(
        dims   =[ 48, 96, 192, 384],
        depths =[ 2, 4, 12, 4],
        ks = [3, 3, 7, 7],
        type =["repdw", "repdw", "repdw", "repsa"],
        act_layer=nn.GELU,
        mlp_ratio=3, 
        final_feature_dim=1280, 
        distillation=False,
     **kwargs   
    )
    # reparameterize(model)
    return model  

@register_model
def revit_sa2(pretrained=False, num_classes = 1000, distillation=False, **kwargs):
    model=REViT(
        dims   =[ 48, 96, 192, 384],
        depths =[ 2, 4, 12, 4],
        ks = [3, 3, 7, 7],
        type =["repdw", "repdw", "repsa", "repsa"],
        act_layer=nn.GELU,
        mlp_ratio=3, 
        final_feature_dim=1280, 
        distillation=False,
     **kwargs   
    )
    # reparameterize(model)
    return model  

@register_model
def revit_ma2(pretrained=False, num_classes = 1000, distillation=False, **kwargs):
    model=REViT(
        dims   =[ 64, 128, 256, 512],
        depths =[ 2, 4, 12, 4],
        ks = [3, 3, 7, 7],
        type =["repdw", "repdw", "repsa", "repsa"],
        act_layer=nn.GELU,
        mlp_ratio=3, 
        final_feature_dim=1280, 
        distillation=False,
     **kwargs   
    )
    # reparameterize(model)
    return model     

# @register_model
# def fastformer_m2(pretrained=True, pretrained_cfg=None, pretrained_cfg_overlay=None, num_classes = 1000, distillation=False, **kwargs):
#     model=FastFormer(
#         dims=[ 32, 64, 160, 256],
#         depths=[ 1, 1, 5, 5],
#         type=['conv', 'conv', 'conv', 'conv'],
#         ks = [3, 3, 3, 3],
#         patch_size=4, 
#         mlp_ratio=2, 
#         act_layer=nn.GELU,
#         final_feature_dim=[1024, 1280], 
#         distillation=False,
#      **kwargs   
#     )
#     # reparameterize(model)
#     return model

# @register_model
# def regnet_t1(pretrained=True, pretrained_cfg=None, pretrained_cfg_overlay=None, num_classes = 1000, distillation=False, **kwargs):
#     model=RegNeXt(
#         dims=[ 48, 96, 192, 384],
#         depths=[ 2, 2, 12, 2],
#         patch_size=4, 
#         mlp_ratio=2, 
#         act_layer=nn.GELU,
#         final_feature_dim= 1280, 
#         distillation=False,
#      **kwargs   
#     )
#     # reparameterize(model)
#     return model


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
