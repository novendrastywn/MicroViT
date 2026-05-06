import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import SqueezeExcite, to_2tuple
from timm.models.vision_transformer import trunc_normal_
from timm.models import register_model

from torch.nn.modules.batchnorm import _BatchNorm

try:
    # mmdet 2
    from mmdet.utils import get_root_logger
    from mmdet.models.builder import BACKBONES as MODELS
    from mmcv.runner import load_checkpoint
    mmdet_lib=2
except ImportError:
    # mmdet 3
    from mmdet.registry import MODELS
    from mmengine.logging import MMLogger
    from mmengine.runner.checkpoint import load_checkpoint
    mmdet_lib=3

class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def reparam(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(0), w.shape[2:], 
                            stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, 
                            groups=self.c.groups, device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class BN_Linear(nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def reparam(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class RepConv(nn.Module):
    def __init__(self, inc, ouc, ks=1, stride=1, pad=0, groups=1):
        super().__init__()

        self.conv = nn.Conv2d(inc, ouc, ks, stride, pad, groups=groups)
        self.repconv = nn.Conv2d(inc, ouc, ks//2, stride, pad//2, groups=groups)
        self.bn = nn.BatchNorm2d(ouc)
    
    def forward(self, x):
        xr = self.conv(x) + self.repconv(x) 
        return self.bn(xr)
    
    @torch.no_grad()
    def reparam(self):
        conv = self.conv

        repconv=self.repconv; self.__delattr__('repconv')
        kw, kh = (conv.weight.shape[2]-repconv.weight.shape[2])//2, \
                     (conv.weight.shape[3]-repconv.weight.shape[3])//2
        repconv_w = nn.functional.pad(repconv.weight, [kh,kh,kw,kw])
        repconv_b = repconv.bias         

        conv_w = conv.weight + repconv_w 
        conv_b = conv.bias + repconv_b 

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = conv_w * w[:, None, None, None]
        b = bn.bias + (conv_b - bn.running_mean) * bn.weight / \
                    (bn.running_var + bn.eps)**0.5
        self.__delattr__('bn')

        m = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, 
                      stride=conv.stride, padding=conv.padding, groups=conv.groups)

        m.weight.data.copy_(w)
        m.bias.data.copy_(b)

        return m


class Residual(nn.Module):

    def __init__(self, m, drop=0.):

        super().__init__()
        self.m = m
        self.drop = drop

        if  self.drop > 0:
            self.forward = self.forward_drop
        else:
            self.forward = self.forward_deploy

    def forward_drop(self, x):
        return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                         device=x.device).ge_(self.drop).div(1 - self.drop).detach()

    def forward_deploy(self, x):
        return x + self.m(x)
  
    @torch.no_grad()
    def reparam(self):
        if isinstance(self.m, Conv2d_BN) and isinstance(self.m, nn.Identity):
            m = self.m.reparam()
            assert(m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            return m
        
        elif isinstance(self.m, RepConv):
            m = self.m.reparam()
            # assert(m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            return m

        else:
            return self

class FFN(torch.nn.Module):
    def __init__(self, ed, h, act_layer=nn.GELU):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = act_layer()
        self.pw2 = Conv2d_BN(h, ed)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x
    
class ConvFFN(torch.nn.Module):
    def __init__(self, ed, h, act_layer=nn.GELU):
        super().__init__()
        self.block = nn.Sequential(
            Conv2d_BN(ed, ed, ks=3, stride=1, pad=1, groups=ed),
            Conv2d_BN(ed, h),
            act_layer(),
            Conv2d_BN(h, ed)
        )   

    def forward(self, x):
        return self.block(x)

class Classfier(nn.Module):
    def __init__(self, dim, num_classes, distillation=True):
        super().__init__()
        self.classifier = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.classifier_dist = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()

    def forward(self, x):
        if self.distillation:
            x = self.classifier(x), self.classifier_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.classifier(x)
        return x

    @torch.no_grad()
    def reparam(self):
        classifier = self.classifier.reparam()
        if self.distillation:
            classifier_dist = self.classifier_dist.reparam()
            classifier.weight += classifier_dist.weight
            classifier.bias += classifier_dist.bias
            classifier.weight /= 2
            classifier.bias /= 2
            return classifier
        else:
            return classifier

# class StemLayer(nn.Module):
#     def __init__(self, inc, ouc, ks=3, ps=16, act_layer=nn.ReLU):
#         super().__init__()
#         pad=0 if (ks % 2)==0 else ks//2

#         blocks = math.ceil(ps**0.5)
#         dims = [inc] + [x.item() for x in ouc//2**torch.arange(blocks-1, -1, -1)]
#         stem = [nn.Sequential(
#                 RepConv(dims[i], dims[i+1], ks=ks, stride=2, pad=pad),
#                 act_layer())
#                 for i in range (blocks)]
#         self.stem = nn.Sequential(*stem)
        
#     def forward(self, x):
#         return self.stem(x)

class StemLayer(nn.Module):
    def __init__(self, inc, ouc, ks=3, ps=16, act_layer=nn.ReLU):
        super().__init__()
        pad=0 if (ks % 2)==0 else ks//2

        blocks = math.ceil(ps**0.5)
        dims = [inc] + [x.item() for x in ouc//2**torch.arange(blocks-1, -1, -1)]
        stem = [nn.Sequential(
                RepConv(dims[i], dims[i+1], ks=ks, stride=2, pad=pad),
                act_layer()) if i<(blocks-1) else \
                RepConv(dims[i], dims[i+1], ks=ks, stride=2, pad=pad)
                for i in range (blocks)]
        self.stem = nn.Sequential(*stem)
        
    def forward(self, x):
        return self.stem(x)

class SSHA(nn.Module):
    def __init__(self, dim, qk_dim, pdim, sr=2, dcons=True, inp_group=1):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim
        self.split_index = (qk_dim, qk_dim, pdim, dim-pdim)
        self.pre_norm = nn.GroupNorm(1, dim)
        self.in_proj = Conv2d_BN(dim, qk_dim*2+dim, 3, sr, 1, groups=inp_group)
        if sr > 1:
            self.ups = nn.ConvTranspose2d(dim, dim, sr*(2 if dcons else 1), stride=sr, 
                                          padding= sr//2 if dcons else 0, groups=dim)
        else:
            self.ups = nn.Identity()
            
        self.out_proj = nn.Sequential(nn.GELU(),
                            Conv2d_BN(dim, dim, 1, 1))
        
    def forward(self, x):
        x = self.pre_norm(x) 
        q, k, v, u = self.in_proj(x).split(self.split_index, dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim = -1)
        # print(attn.shape, u.shape)
        B, _, H, W = u.shape
        attn = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
        out  = self.out_proj(self.ups(torch.cat((attn, u), dim=1)))
        # print(out.shape)
        return out
    
class ESHA(nn.Module):
    def __init__(self, dim, pdim, qk_dim=16, sr=1, inp_group=32):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim
        self.split_index = (qk_dim, qk_dim, pdim, dim-pdim)
        self.pre_norm = nn.GroupNorm(1, dim)
        self.in_proj = RepConv(dim, (qk_dim*2)+dim, 3, 1, 1, groups=inp_group)
        self.out_proj = nn.Sequential(nn.GELU(),
                        Conv2d_BN(dim, dim, 1, 1))
        
    def forward(self, x):
        x = self.pre_norm(x) 
        q, k, v, u = self.in_proj(x).split(self.split_index, dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim = -1)
        # print(attn.shape, u.shape)
        B, _, H, W = u.shape
        attn = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
        out  = self.out_proj(torch.cat((attn, u), dim=1))
        # print(out.shape)
        return out

class SDTA(nn.Module):
    def __init__(self, dim, pdim, qk_dim=16):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim
        self.split_index = (qk_dim, qk_dim, pdim, dim-pdim)
        self.pre_norm = nn.GroupNorm(1, dim)
        hid = (qk_dim*2)+dim
        self.in_proj = nn.Sequential(
                        RepConv(dim, dim, 3, 1, 1, groups=dim),
                        Conv2d_BN(dim, hid),
                       )

        # self.in_proj = RepConv(dim, hid, 3, 1, 1)
    

        self.out_proj = nn.Sequential(nn.GELU(),
                        Conv2d_BN(dim, dim, 1, 1))
        
    def forward(self, x):
        x = self.pre_norm(x) 
        q, k, v, u = self.in_proj(x).split(self.split_index, dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim = -1)

        B, _, H, W = u.shape
        attn = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
        out  = self.out_proj(torch.cat((attn, u), dim=1))

        return out


class DSHA(nn.Module):
    def __init__(self, dim, pdim, qk_dim=16):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim
        self.split_index = (qk_dim, qk_dim, pdim, pdim)
        self.pre_norm = nn.GroupNorm(1, dim)
        hid = (qk_dim*2)+(2*pdim)
        self.in_proj = nn.Sequential(
                        Conv2d_BN(dim, hid, 3, 1, 1),
                        # Conv2d_BN(hid, hid, 3, 1, 1, groups=hid)
                       )

        self.out_proj = nn.Sequential(nn.GELU(),
                        Conv2d_BN((2*pdim), dim, 1, 1))
        
    def forward(self, x):
        x = self.pre_norm(x)
        # print(x.shape) 
        q, k, v, u = self.in_proj(x).split(self.split_index, dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim = -1)
        # print(attn.shape, u.shape)
        B, _, H, W = u.shape
        attn = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
        out  = self.out_proj(torch.cat((attn, u), dim=1))
        # print(out.shape)
        return out


class PatchMerging(nn.Module):
    def __init__(self, inc, ouc, ks=3, act_layer=nn.ReLU):
        super().__init__()
        pad=0 if (ks % 2)==0 else ks//2 
        # hidden = int(ouc*4)
        self.token_mix  = nn.Sequential(
                        # Conv2d_BN(inc, ouc, ks=1, stride=1),
                        RepConv(inc, inc, ks=3, stride=2, pad=1, groups=inc),
                        act_layer(),
                        Conv2d_BN(inc, ouc, ks=1, stride=1)
                        )
        self.channel_mix = Residual(nn.Sequential(
                        Conv2d_BN(ouc, ouc*2, ks=1, stride=1, pad=0),
                        act_layer(),
                        Conv2d_BN(ouc*2, ouc, ks=1, stride=1, pad=0))
                        )
    def forward(self, x):
        return self.channel_mix(self.token_mix(x))

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio, att_cr, type, act_layer=nn.ReLU):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio) 
        pdim = int(dim * att_cr)

        if type == 'f':
            self.block = Residual(nn.Sequential(
                            RepConv(dim, dim, 3, 1, 1, groups=1),
                            act_layer(),
                            Conv2d_BN(dim, dim, ks=1, stride=1, pad=0)))
        else:
            if type == 'c':
                spatial_mix = RepConv(dim, dim, 3, 1, 1, groups=dim)
            elif type== 'esha':
                spatial_mix = ESHA(dim, pdim)
            elif type== 'sdta':
                spatial_mix = SDTA(dim, pdim)

            self.block  = nn.Sequential(
                            Residual(spatial_mix),
                            Residual(FFN(dim, hidden_dim, act_layer))
                            )
    def forward(self, x):
        return self.block(x)
    
class Stage(nn.Module):
    def __init__(self, dim, depth, mlp_ratio, att_cr, type, act_layer=nn.ReLU):
        super().__init__()
        block = [
                Block(dim,
                      mlp_ratio = mlp_ratio,
                      att_cr  = att_cr, 
                      act_layer = act_layer,
                      type = type 
                      ) for i in range (depth)
            ]

        self.blocks = nn.Sequential(*block)
    def forward(self, x):
        return self.blocks(x)
    
class MicroViTv2(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
            dims   = [ 128, 256, 320],
            depths = [ 2, 4, 4],
            type   = [ 'f', 'c', 'a'],
            attn_cr=[ 0, 0, 0.25],
            patch_size = 32,
            mlp_ratio=2, 
            act_layer=nn.ReLU,
            final_feature=1024, 
            distillation=False, 
            pretrained=None, 
            **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.final_feature_dim = final_feature

        if not isinstance(depths, (list, tuple)):
            depths = [depths] # it means the model has only one stage
        if not isinstance(dims, (list, tuple)):
            dims = [dims]
        
        num_stage = len(depths)
        self.num_stage = num_stage

        stages = []
        stages.append(StemLayer(in_chans, dims[0], ps=patch_size, act_layer=act_layer))
        n_stage=0
        self.indice=[]

        for i_stage in range(num_stage):
            stage = Stage(
                    dim=dims[i_stage],
                    depth=depths[i_stage],  
                    mlp_ratio=mlp_ratio,
                    att_cr=attn_cr[i_stage], 
                    act_layer=act_layer,
                    type=type[i_stage]
            )
            stages.append(stage)
            n_stage+=1
            self.indice.append(n_stage)
            if i_stage < (num_stage-1):
                pre_patch= nn.Sequential(
                    Residual(RepConv(dims[i_stage], dims[i_stage], 3, 1, 1, groups=dims[i_stage])),
                    Residual(FFN(dims[i_stage], dims[i_stage]*2, act_layer=act_layer)),
                )
                # patch_merging=PatchMerging(dims[i_stage], dims[i_stage+1], act_layer=act_layer)
                patch_merging=RepConv(dims[i_stage], dims[i_stage+1], 3, 2, 1)
                pos_patch= nn.Sequential(
                    Residual(RepConv(dims[i_stage+1], dims[i_stage+1], 3, 1, 1, groups=dims[i_stage+1])),
                    Residual(FFN(dims[i_stage+1], dims[i_stage+1]*2, act_layer=act_layer)),
                )
                stages.append(pre_patch)
                stages.append(patch_merging)
                stages.append(pos_patch)
                n_stage+=3

        self.stages = nn.Sequential(*stages)

        if pretrained:
            print("this is pretrained:", pretrained)
            self.init_weights(pretrained)
        else:
            self.apply(self.cls_init_weights)
        
        # self = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
        # self.train()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            
            if mmdet_lib==2 :
                # mmdet 2
                logger = get_root_logger()
                load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
            elif mmdet_lib==3 :
                # mmdet 3
                logger = MMLogger.get_current_instance()
                load_checkpoint(self, pretrained, strict=False, logger=logger)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(MicroViTv2, self).train(mode)
        if mode:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
            

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_feature(self, x):
        out=[]
        for i, f in enumerate(self.stages):
            x = f(x)
            if i in self.indice:
                # print(x.shape, i)
                out.append(x)

        return out

    def forward(self, x):
        x = self.forward_feature(x)
        return x
    

@MODELS.register_module()
class microvitv2_3(MicroViTv2):
    def __init__(self, **kwargs):
        super().__init__(
            dims   = [ 192, 384, 448],
            depths = [ 3, 7, 6],
            type   = [ 'c', 'c', 'sdta'],
            attn_cr = [ 0, 0, 0.25],
            patch_size = 16,
            act_layer = nn.SiLU,
            final_feature=None,
            pretrained='/home/dsdl-4090/Documents/dsdl_ssd/MicroViTv2/output/microvitv2_3_sdta_/best_checkpoint.pth',
            **kwargs
            )
