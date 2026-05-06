import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.layers import SqueezeExcite
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math

class Conv2d_BN(nn.Sequential):
    def __init__(self, inc, ouc, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            inc, ouc, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(ouc))
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

class GroupNorm(torch.nn.GroupNorm):
    def __init__(self, dim, **kwargs):
        super().__init__(1, dim, **kwargs)

class BN_Linear(nn.Sequential):
    def __init__(self, inc, ouc, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(inc))
        self.add_module('l', torch.nn.Linear(inc, ouc, bias=bias))
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

    def __init__(self, m, dim, drop=0., ls_init=0):

        super().__init__()
        self.m = m
        self.drop = drop
        # self.ls = nn.Parameter(ls_init*torch.ones(dim, 1, 1), requires_grad=True)

        if  self.drop > 0:
            self.forward = self.forward_drop
            self.drop_path = DropPath(drop)
        else:
            self.forward = self.forward_deploy

    def forward_drop(self, x):
        return x + self.drop_path(self.m(x))  # torch.rand(x.size(0), 1, 1, 1,
                                         # device=x.device).ge_(self.drop).div(1 - self.drop).detach()

    def forward_deploy(self, x):
        return x + self.m(x)

        
    @torch.no_grad()
    def reparam(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.reparam()
            assert(m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            return m

        else:
            return self

class SSEmodule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 1, 1)
    
    def forward(self, x):
        gca = F.adaptive_avg_pool2d(x, 1)
        exc = F.sigmoid(self.conv(gca))
        return exc*x

class StrideConv(nn.Module):
    def __init__(self, inc, ouc, ks=3, s=16, act_layer=nn.SiLU):
        super().__init__()
        pad=0 if (ks % 2)==0 else ks//2
        blocks = 1 if s==2 else math.ceil(s**0.5) 
        dims = [inc] + [x.item() for x in ouc//2**torch.arange(blocks-1, -1, -1)]
        stem = [nn.Sequential(
                Conv2d_BN(dims[i], dims[i+1], ks=ks, stride=2, pad=pad),
                act_layer() if i < (blocks-1) else nn.Identity())
                for i in range (blocks)]
        self.stem = nn.Sequential(*stem)
        
    def forward(self, x):
        return self.stem(x)

class PatchEmbedding(nn.Module):
    """ Channel Attention Pacth Embedding"""
    def __init__(self, inc=3, ouc=768, ks=3, s=4, se=0, act_layer=nn.SiLU):
        super().__init__()

        if s>2:
            self.conv_proj = StrideConv(inc, ouc, ks=ks, s=s, act_layer=act_layer)
        else:
            hid_dim = int(inc * 4)
            self.conv_proj = nn.Sequential(
                            Conv2d_BN(inc, hid_dim, 1, 1, 0, ),
                            act_layer(),
                            Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim,), 
                            Conv2d_BN(hid_dim, ouc, 1, 1, 0,))

        self.se = SSEmodule(ouc) if se !=0 else nn.Identity()
    def forward(self, x):
        x = self.conv_proj(x) # Convolutional Patch Embedding
        x = self.se(x)
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

class LocalAggregation(nn.Module):
    def __init__(self, dim, ks=3):
        super().__init__()
        pad = ks//2
        self.conv = nn.Conv2d(dim, dim, ks, 1, pad, groups=dim)
        self.repconv = nn.Conv2d(dim, dim, ks//2, 1, pad//2, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
    
    def forward(self, x):
        xr = self.conv(x) + self.repconv(x) 
        return self.bn(xr)
    
    @torch.no_grad()
    def reparam(self):
        conv = self.conv

        repconv=self.repconv; self.__delattr__('repconv')
        kw, kh = (conv.weight.shape[2]-repconv.weight.shape[2])//2, \
                     (conv.weight.shape[3]-repconv.weight.shape[3])//2
        conv_w = conv.weight + F.pad(repconv.weight, [kh,kh,kw,kw])
        conv_b = conv.bias + repconv.bias          

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = conv_w * w[:, None, None, None]
        b = bn.bias + (conv_b - bn.running_mean) * bn.weight / \
                    (bn.running_var + bn.eps)**0.5
        self.__delattr__('bn')

        dim = conv.in_channels
        ks = conv.kernel_size
        pad = conv.padding
        m = nn.Conv2d(dim, dim, ks, 1, pad, groups=dim)

        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        self.__delattr__('conv')
        return m

class SCAtt(nn.Module):

    def __init__(self, ks=7):
        super().__init__()
        assert ks in {3, 7}, "kernel size must be 3 or 7"
        pad = 3 if ks == 7 else 1
        self.conv = nn.Conv2d(2, 1, ks, padding=pad, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        att = torch.cat([torch.mean(x, 1, keepdim=True), 
                         torch.max(x, 1, keepdim=True)[0]], 1)
        return x * self.act(self.conv(att))

class SSHAtt(nn.Module):
    """Sparse Single Head Attention"""
    def __init__(self, dim, att_ratio=0.25, sr_ratio=1, qkdim=16, **kwargs):
        super().__init__()
        self.scale = qkdim**-0.5
        self.att_dim = int (dim * att_ratio)
        self.split_index = (qkdim, qkdim, self.att_dim)
        # self.sparse_proj = nn.Sequential(
        #                     nn.AvgPool2d(3, stride=sr_ratio, padding=1),
        #                     Conv2d_BN(dim, 2*qkdim + self.att_dim, 1, 1))
        self.sparse_proj = Conv2d_BN(dim, 2*qkdim + self.att_dim, 1, sr_ratio)
        
        if sr_ratio>1: 
            self.local_prop  = nn.ConvTranspose2d(self.att_dim, self.att_dim, 
                                                  sr_ratio, sr_ratio, groups=self.att_dim)
        else:
            self.local_prop = nn.Identity()

    def forward(self, x):
        
        x = self.sparse_proj(x)
        B, C, H, W = x.shape
        q, k, v = torch.split(x, self.split_index, dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim = -1)
        attn = (v @ attn.transpose(-2, -1)).reshape(B, self.att_dim, H, W)

        return self.local_prop(attn)

class ParallelMixer(nn.Module):
    def __init__(self, dim, att_ratio=0.25, sr_ratio=1, **kwargs):
        super().__init__()
        self.att_dim = int(dim * att_ratio)
        self.loa  = LocalAggregation(dim, 3)
        self.csa  = SCAtt(ks=3)
        self.ssha = SSHAtt(dim, att_ratio, sr_ratio)
        self.proj = Conv2d_BN(self.att_dim + dim, dim)
        # self.act = act_layer()

    def forward(self, x):
        x   = self.loa(x)
        att = self.ssha(x)
        ctt = self.csa(x)
        # print(att.shape, ctt.shape)
        x = self.proj(torch.cat((att, ctt), dim=1))
        return x
    
class MlpHead(nn.Module):
    """ MLP classification head
    """
    def __init__(self, dim, num_classes=1000, mlp_ratio=4, act_layer=nn.GELU):
        super().__init__()
        hidden_features = min(int(mlp_ratio * dim), 1280)
        self.fc1 = BN_Linear(dim, hidden_features)
        self.act = act_layer()
        self.fc2 = BN_Linear(hidden_features, num_classes)
  
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class ConvFFN(nn.Module):
    def __init__(self, inc, hidd=None, ouc=None, act_layer=nn.SiLU):
        super().__init__()
        ouc = ouc or inc
        hidd = hidd or inc
        self.dwc = Conv2d_BN(inc, inc, 3, 1, 1, groups=inc)
        self.ffn = nn.Sequential(
                    Conv2d_BN(inc, hidd, 1, 1),
                    act_layer(),
                    Conv2d_BN(hidd, ouc,1, 1))
        
    def forward(self, x):
        x = self.dwc(x)
        x = self.ffn(x)
        return x

class ParFormerBlock(nn.Module):

    def __init__(self, dim, att_ratio=0.25, sr_ratio=1, act_layer=nn.SiLU, drop=0.):
        super().__init__()           
        hid = int(dim * 2)
        ffn_pra = ConvFFN(inc=dim, hidd=hid, act_layer=act_layer)
        ffn_pre = ConvFFN(inc=dim, hidd=hid, act_layer=act_layer)
        self.ffn_pre = Residual(ffn_pre, dim, drop, 1)
        self.token_mix = Residual(ParallelMixer(dim, att_ratio, sr_ratio), 
                                  dim, drop, 1)
        self.ffn_pra = Residual(ffn_pra, dim, drop, 1)

    def forward(self, x):

        return self.ffn_pra(self.token_mix(self.ffn_pre(x)))

class ParFormer(nn.Module):
    def __init__(self, img_size=224, num_classes=1000, embed_dims=[128, 384, 512], embed_stride=[16, 2, 2], 
                 depths=[3, 9, 3], att_ratio=[0.25, 0.25, 0.25], sr_ratio=[8, 2, 2], num_stages=3, 
                 drop_rate=0.05, cape=[1, 1, 1], act_layer=nn.SiLU, distillation=False,
                 head_dropout=0.0, head_init_scale=1.0, pre_head='avg', **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.downsamplings=nn.ModuleList()
        self.stages=nn.ModuleList()
        for i in range(num_stages):
            patch_embed = PatchEmbedding(3 if i == 0 else embed_dims[i - 1], embed_dims[i],
                                         2 if embed_stride[i] == 3 else 3, embed_stride[i], 
                                         cape[i], act_layer)
            self.downsamplings.append(patch_embed) 
            
            block = nn.Sequential(
                *[ParFormerBlock(dim=embed_dims[i],
                                 att_ratio=att_ratio[i], 
                                 sr_ratio=sr_ratio[i],  
                                 drop=dpr[cur + j], 
                                 act_layer=act_layer)
                for j in range(depths[i])])
            self.stages.append(block)
            
            cur += depths[i]

        # classification head
        self.head = Classfier(embed_dims[-1], num_classes, distillation)

        if pre_head is "avg":
            self.pre_head = nn.AdaptiveAvgPool2d(1)
        elif pre_head is "gdc":
            import math
            ks = 4 if math.prod(embed_stride) == 64 else 7
            self.pre_head = Conv2d_BN(embed_dims[-1], embed_dims[-1], 4, 1, groups=embed_dims[-1])
        else:
            raise ValueError("pre_head must be 'avg' or 'gd'")
    
        # print(self)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = MlpHead(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            x = self.downsamplings[i](x)
            x = self.stages[i](x)
        return self.pre_head(x).flatten(1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

