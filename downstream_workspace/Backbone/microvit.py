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
        if isinstance(self.m, Conv2d_BN) or isinstance(self.m, nn.Identity):
            m = self.m.reparam()
            assert(m.groups == m.in_channels)
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

class StemLayer(nn.Module):
    def __init__(self, inc, ouc, ks=3, ps=16, act_layer=nn.ReLU):
        super().__init__()
        pad=0 if (ks % 2)==0 else ks//2

        blocks = math.ceil(ps**0.5)
        dims = [inc] + [x.item() for x in ouc//2**torch.arange(blocks-1, -1, -1)]
        stem = [nn.Sequential(
                Conv2d_BN(dims[i], dims[i+1], ks=ks, stride=2),
                act_layer())
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
    def __init__(self, dim, qk_dim, pdim, sr=2, inp_group=1):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim
        self.split_index = (qk_dim, qk_dim, pdim, dim-pdim)
        self.pre_norm = nn.GroupNorm(1, dim)
        self.in_proj = Conv2d_BN(dim, (qk_dim*2)+dim, 3, 1, 1, groups=inp_group)
        if sr > 1:
            self.k = Conv2d_BN(qk_dim, qk_dim, sr, sr, groups=qk_dim)
            self.v = Conv2d_BN(pdim, pdim, sr, sr, groups=pdim)
        else:
            self.k = nn.Identity()
            self.v = nn.Identity()
        self.out_proj = nn.Sequential(nn.ReLU(),
                        Conv2d_BN(dim, dim, 1, 1))
        
    def forward(self, x):
        x = self.pre_norm(x) 
        q, k, v, u = self.in_proj(x).split(self.split_index, dim=1)
        q, k, v = q.flatten(2), self.k(k).flatten(2), self.v(v).flatten(2)
        
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim = -1)
        # print(attn.shape, u.shape)
        B, _, H, W = u.shape
        attn = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
        out  = self.out_proj(torch.cat((attn, u), dim=1))
        return out

class PatchMerging(nn.Module):
    def __init__(self, inc, ouc, ks=3, act_layer=nn.ReLU):
        super().__init__()
        pad=0 if (ks % 2)==0 else ks//2 
        # hidden = int(ouc*4)
        self.token_mix  = nn.Sequential(
                        # Conv2d_BN(inc, inc*2, ks=1, stride=1),
                        Conv2d_BN(inc, inc, ks=3, stride=2, pad=1, groups=inc),
                        act_layer(),
                        Conv2d_BN(inc, ouc, ks=1, stride=1)
                        )
        self.channel_mix = Residual(nn.Sequential(
                        Conv2d_BN(ouc, ouc*2, ks=1, stride=1, pad=0),
                        act_layer(),
                        Conv2d_BN(ouc*2, ouc, ks=1, stride=1, pad=0))
                        )
    def forward(self, x):
        #  print(x.shape)
        return self.channel_mix(self.token_mix(x))

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio, qk_dim, att_cr, att_sr, att_ipg,  type, act_layer=nn.ReLU):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        pdim = int(dim * att_cr)

        if type == 'c':
            spatial_mix = Conv2d_BN(dim, dim, 3, 1, 1, groups=dim)
            self.block  = nn.Sequential(
                        Residual(spatial_mix),
                        Residual(FFN(dim, hidden_dim, act_layer))
                        )
        elif type== 'a':
            spatial_mix = ESHA(dim, qk_dim, pdim, sr=att_sr, 
                               inp_group=att_ipg)
            self.block  = nn.Sequential(
                        Residual(spatial_mix),
                        Residual(FFN(dim, hidden_dim, act_layer))
                        )

    def forward(self, x):
        return self.block(x)
    
class Stage(nn.Module):
    def __init__(self, dim, depth, mlp_ratio, qk_dim, att_cr, att_sr, att_ipg, type, act_layer=nn.ReLU):
        super().__init__()
        block = [
                Block(dim,
                      mlp_ratio = mlp_ratio,
                      qk_dim  = qk_dim,
                      att_sr  = att_sr,
                      att_cr  = att_cr, 
                      att_ipg = att_ipg,
                      act_layer = act_layer,
                      type = type 
                      ) for i in range (depth)
            ]

        self.blocks = nn.Sequential(*block)
    def forward(self, x):
        return self.blocks(x)
    
class MicroViT(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 dims=[  48, 96, 192, 384],
                 depths=[ 2, 2, 2, 2],
                 type=[ 'c', 'c', 'a', 'a'],
                 qk_dim = [16, 16, 16, 16],
                 attn_sr=[0, 0, 2, 2],
                 attn_cr=[ 0, 0, 0.25, 0.25],
                 attn_ipg=[ 0, 0, 32, 32],
                 patch_size=4, 
                 mlp_ratio=2, 
                 act_layer=nn.ReLU,
                 final_feature=1024,
                 pretrained=None, 
                 distillation=False, **kwargs):
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
                    qk_dim = qk_dim[i_stage],
                    att_cr=attn_cr[i_stage], 
                    att_ipg=attn_ipg[i_stage],
                    att_sr=attn_sr[i_stage],
                    act_layer=act_layer,
                    type=type[i_stage]
            )
            stages.append(stage)
            n_stage+=1
            self.indice.append(n_stage)
            if i_stage < (num_stage-1):
                # pre_patch= nn.Sequential(
                #     Residual(Conv2d_BN(dims[i_stage], dims[i_stage], 3, 1, 1, groups=dims[i_stage])),
                #     Residual(FFN(dims[i_stage], dims[i_stage]*2, act_layer=act_layer)),
                # )
                patch_merging=PatchMerging(dims[i_stage], dims[i_stage+1], act_layer=act_layer)
                pos_patch= nn.Sequential(
                    Residual(Conv2d_BN(dims[i_stage+1], dims[i_stage+1], 3, 1, 1, groups=dims[i_stage+1])),
                    Residual(FFN(dims[i_stage+1], dims[i_stage+1]*2, act_layer=act_layer)),
                )
                # stages.append(pre_patch)
                stages.append(patch_merging)
                n_stage+=1
                stages.append(pos_patch)
                n_stage+=1

        self.stages = nn.Sequential(*stages)
        # print(self.indice)
       
        if pretrained:
            print("this is pretrained:", pretrained)
            self.init_weights(pretrained)
        else:
            self.apply(self.cls_init_weights)
        
        self = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
        self.train()

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
        super(MicroViT, self).train(mode)
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
class microvit_3(MicroViT):
    def __init__(self, **kwargs):
        super().__init__(
            dims=[ 192, 384, 512],
            depths=[ 3, 6, 6],
            type=[ 'c', 'c', 'a'],
            qk_dim = [0, 0, 16],
            attn_sr=[ 0, 0, 1],
            attn_ipg=[ 0, 0, 32],
            attn_cr=[ 0, 0, 0.215],
            mlp_ratio=2,
            patch_size=16,
            final_feature=None,
            act_layer=nn.GELU,
            pretrained='/home/dsdl-4090-ai-server/Documents/REViTWorkspace/checkpoints/final_version/microvit_3/microvit_3_old/model_best.pth.tar',
            **kwargs
            )
        # reparameterize(model)
        # return model