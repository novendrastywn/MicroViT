import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import SqueezeExcite
from timm.models.layers import to_2tuple
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

class GroupNorm(torch.nn.GroupNorm):
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

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
        if self.training and self.drop > 0:
            self.forward = self.forward_train
        else:
            self.forward = self.forward_deploy

    def forward_train(self, x):
        return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                         device=x.device).ge_(self.drop).div(1 - self.drop).detach()

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
        elif isinstance(self.m, RepDWConv):
            m = self.m.reparam()
            assert(m.conv.groups == m.conv.in_channels)
            ("Identity RepMSDWConv Reparam")
            kw, kh = (m.conv.weight.shape[2]-1)//2, \
                     (m.conv.weight.shape[3]-1)//2
            identity = torch.ones(m.conv.weight.shape[0], m.conv.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [kh,kh,kw,kw])
            m.conv.weight += identity.to(m.conv.weight.device)
            return m
        else:
            return self        

class FFN(torch.nn.Module):
    def __init__(self, ed, h, act_layer=nn.GELU):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = act_layer()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

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

class RepConv(nn.Module):
    def __init__(self, inc, ouc, ks=1, stride=1, pad=0, groups=1):
        super().__init__()

        self.conv = nn.Conv2d(inc, ouc, ks, stride, pad, groups=groups)
        self.repconv = nn.Conv2d(inc, ouc, ks//2, stride, pad//2, groups=groups)
        self.repconvkx1 = nn.Conv2d(inc, ouc, kernel_size=(ks, 1), 
                                    stride=stride, padding=(pad, 0),
                                    groups=groups)
            
        self.repconv1xk = nn.Conv2d(inc, ouc, kernel_size=(1, ks), 
                                    stride=stride, padding=(0, pad),
                                    groups=groups)

        self.bn = nn.BatchNorm2d(ouc)
    
    def forward(self, x):
        xr = self.conv(x) + self.repconv(x) + self.repconvkx1(x) + self.repconv1xk(x)
        return self.bn(xr)
    
    def forward_deploy(self, x):
        return self.conv(x)
    
    @torch.no_grad()
    def reparam(self):
        conv = self.conv

        repconv=self.repconv; self.__delattr__('repconv')
        kw, kh = (conv.weight.shape[2]-repconv.weight.shape[2])//2, \
                     (conv.weight.shape[3]-repconv.weight.shape[3])//2
        repconv_w = nn.functional.pad(repconv.weight, [kh,kh,kw,kw])
        repconv_b = repconv.bias         

        repconv = self.repconvkx1; self.__delattr__('repconvkx1')
        kw, kh  = (conv.weight.shape[2]-repconv.weight.shape[2])//2, \
                        (conv.weight.shape[3]-repconv.weight.shape[3])//2
        repconv_w += nn.functional.pad(repconv.weight, [kh,kh,kw,kw])
        repconv_b += repconv.bias

        repconv = self.repconv1xk; self.__delattr__('repconv1xk')
        kw, kh  = (conv.weight.shape[2]-repconv.weight.shape[2])//2, \
                        (conv.weight.shape[3]-repconv.weight.shape[3])//2
        repconv_w += nn.functional.pad(repconv.weight, [kh,kh,kw,kw])
        repconv_b += repconv.bias

        final_conv_w = conv.weight + repconv_w 
        final_conv_b = conv.bias + repconv_b 

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
                    (bn.running_var + bn.eps)**0.5
        self.__delattr__('bn')

        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)

        self.forward = self.forward_deploy
        self.conv = conv
        return self

class RepDWConv(nn.Module):
    def __init__(self, dim, ks=1, stride=1, pad=0):
        super().__init__()
        repdim = dim//4
        self.repdim = repdim
        self.conv = nn.Conv2d(dim, dim, ks, stride, pad, groups=dim)
        kwargs = {"in_channels": repdim, "out_channels": repdim, "groups": repdim}
        self.repconvsxs = nn.Conv2d(kernel_size=max(ks//2, 3),   stride=stride, padding=max(pad//2, 1),   **kwargs)
        self.repconvkx1 = nn.Conv2d(kernel_size=(ks, 1), stride=stride, padding=(pad, 0), **kwargs)
        self.repconv1xk = nn.Conv2d(kernel_size=(1, ks), stride=stride, padding=(0, pad), **kwargs)
        self.repconvkx3 = nn.Conv2d(kernel_size=(ks, 3), stride=stride, padding=(pad, 1), **kwargs)
        self.repconv3xk = nn.Conv2d(kernel_size=(3, ks), stride=stride, padding=(1, pad), **kwargs)
        self.stride = stride
        if stride>1:
            self.repconvsxs_s2 = nn.Conv2d(kernel_size=ks//2,   stride=stride, padding=pad//2,   **kwargs)
        else:
            self.repconvsxs_s2 = nn.Identity()

        self.bn = nn.BatchNorm2d(dim)
    
    def forward(self, x):
        xr = self.conv(x) 
        x1, x2, x3, x4 = x.chunk(4, dim=1)
        # print(self.repconvsxs(x1).shape, 
        #     (self.repconvkx1(x2)+self.repconv1xk(x2)).shape,
        #     (self.repconvkx3(x3)+self.repconv3xk(x3)).shape,
        #     self.repconvsxs_s2(x4).shape)
        xr +=torch.cat((self.repconvsxs(x1), 
            self.repconvkx1(x2)+self.repconv1xk(x2),
            self.repconvkx3(x3)+self.repconv3xk(x3),
            self.repconvsxs_s2(x4)), dim=1)
        return self.bn(xr)
    
    def forward_deploy(self, x):
        return self.conv(x)
    
    @torch.no_grad()
    def reparam(self):
        conv = self.conv

        repconv=self.repconvsxs; self.__delattr__('repconvsxs')
        kw, kh = (conv.weight.shape[2]-repconv.weight.shape[2])//2, \
                     (conv.weight.shape[3]-repconv.weight.shape[3])//2
        repconvs_w = nn.functional.pad(repconv.weight, [kh,kh,kw,kw])
        repconvs_b = repconv.bias         

        repconv = self.repconvkx1; self.__delattr__('repconvkx1')
        kw, kh  = (conv.weight.shape[2]-repconv.weight.shape[2])//2, \
                        (conv.weight.shape[3]-repconv.weight.shape[3])//2
        repconv1_w = nn.functional.pad(repconv.weight, [kh,kh,kw,kw])
        repconv1_b = repconv.bias

        repconv = self.repconv1xk; self.__delattr__('repconv1xk')
        kw, kh  = (conv.weight.shape[2]-repconv.weight.shape[2])//2, \
                        (conv.weight.shape[3]-repconv.weight.shape[3])//2
        repconv1_w += nn.functional.pad(repconv.weight, [kh,kh,kw,kw])
        repconv1_b += repconv.bias

        repconv = self.repconvkx3; self.__delattr__('repconvkx3')
        kw, kh  = (conv.weight.shape[2]-repconv.weight.shape[2])//2, \
                        (conv.weight.shape[3]-repconv.weight.shape[3])//2
        repconv3_w = nn.functional.pad(repconv.weight, [kh,kh,kw,kw])
        repconv3_b = repconv.bias

        repconv = self.repconv3xk; self.__delattr__('repconv3xk')
        kw, kh  = (conv.weight.shape[2]-repconv.weight.shape[2])//2, \
                        (conv.weight.shape[3]-repconv.weight.shape[3])//2
        repconv3_w += nn.functional.pad(repconv.weight, [kh,kh,kw,kw])
        repconv3_b += repconv.bias

        if self.stride > 1:
            repconv=self.repconvsxs_s2; self.__delattr__('repconvsxs_s2')
            kw, kh = (conv.weight.shape[2]-repconv.weight.shape[2])//2, \
                        (conv.weight.shape[3]-repconv.weight.shape[3])//2
            identity_w = nn.functional.pad(repconv.weight, [kh,kh,kw,kw])
            identity_b = repconv.bias 
        else:
            self.__delattr__('repconvsxs_s2')
            identity = torch.ones(repconv.weight.shape[0], repconv.weight.shape[1], 1, 1)
            kw, kh = (conv.weight.shape[2]-1)//2, (conv.weight.shape[3]-1)//2
            identity_w = torch.nn.functional.pad(identity, [kh,kh,kw,kw])
            identity_b = torch.zeros(repconv.weight.shape[0])

        repconv_w = torch.cat((repconvs_w, repconv1_w, repconv3_w, identity_w), dim=0)
        repconv_b = torch.cat((repconvs_b, repconv1_b, repconv3_b, identity_b), dim=0)

        final_conv_w = conv.weight + repconv_w 
        final_conv_b = conv.bias + repconv_b 

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
                    (bn.running_var + bn.eps)**0.5
        self.__delattr__('bn')

        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)

        self.forward = self.forward_deploy
        self.conv = conv
        return self

class StemLayer(nn.Module):
    def __init__(self, inc, ouc, ks=3, ps=16, act_layer=nn.ReLU):
        super().__init__()
        pad=0 if (ks % 2)==0 else ks//2

        blocks = math.ceil(ps**0.5)
        dims = [inc] + [x.item() for x in ouc//2**torch.arange(blocks-1, -1, -1)]
        stem = [nn.Sequential(
                RepConv(dims[i], dims[i+1], ks=ks, stride=2, pad=pad),
                act_layer() if i < (blocks-1) else nn.Identity())
                for i in range (blocks)]
        self.stem = nn.Sequential(*stem)
        
    def forward(self, x):
        return self.stem(x)
    
class PatchMerging(nn.Module):
    def __init__(self, inc, ouc, ks=7, act_layer=nn.ReLU):
        super().__init__()
        pad=0 if (ks % 2)==0 else ks//2 

        self.spatial = nn.Sequential(
                        RepDWConv(inc, ks=ks, stride=2, pad=pad),
                        act_layer(),
                        Conv2d_BN(inc, ouc, ks=1, stride=1)
                        )
        self.channel = Residual(FFN(ouc, ouc*2, act_layer))
        
    def forward(self, x):
        return self.channel(self.spatial(x))
        

class RepMHSA(torch.nn.Module):
    """Structural Reparameterization Single-Head Self-Attention"""
    def __init__(self, dim, ratio=0.4, ks=3, act_layer=nn.ReLU):
        super().__init__()

        self.n_head = 6
        self.ratio  = ratio
        self.qk_dim = 32
        self.v_dim  = int( dim * self.ratio) 
        self.att_dim = self.n_head * self.v_dim
        self.split_idx = (self.qk_dim, self.qk_dim, self.v_dim)
        self.head_dim = self.qk_dim + self.qk_dim + self.v_dim

        self.scale     = self.qk_dim ** -0.5
        proj_dim = 2 * self.qk_dim * self.n_head + self.v_dim  * self.n_head
        self.qkv_proj  = nn.Sequential(
                         RepDWConv(dim, ks, 1, ks//2),
                         Conv2d_BN(dim, proj_dim, 1, 1)
                         )
        self.out_proj  = Conv2d_BN(self.att_dim, dim, 1, 1)
        
    def forward(self, x):
        B, _, H, W = x.shape

        qkv = self.qkv_proj(x).reshape(B, self.n_head, self.head_dim, -1)
        q, k, v = qkv.permute(0, 1, 3, 2).split(self.split_idx, dim=-1)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim = -1)
        attn = torch.matmul(attn, v).transpose(-2, -1).reshape(B, -1, H, W)
        x = self.out_proj(attn)
        # print(x.shape)
        return x

class RepSA(torch.nn.Module):
    """Structural Reparameterization Single-Head Self-Attention"""
    def __init__(self, dim, qk_dim=16, ks=3, act_layer=nn.ReLU):
        super().__init__()
        self.scale     = qk_dim ** -0.5
        self.att_dim   = int(dim*0.215)
        self.split_idx = (qk_dim, qk_dim, self.att_dim)
        self.conv_proj = RepDWConv(dim, ks, 1, ks//2)
        self.qkv_proj  = Conv2d_BN(dim, qk_dim*2 + self.att_dim, 1, 1)
        self.out_proj  = nn.Sequential(act_layer(),
                            Conv2d_BN(dim+self.att_dim, dim, 1, 1))
        
    def forward(self, x):
        B, _, H, W = x.shape

        u = self.conv_proj(x)
        q, k, v = self.qkv_proj(u).split(self.split_idx, dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        
        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attn = attn.softmax(dim = -1)

        attn = torch.matmul(v, attn.transpose(-2, -1)).reshape(B, -1, H, W)

        x = self.out_proj(torch.cat((attn, u), dim=1))
        return x

class Block(nn.Module):
    def __init__(self, dim, cr, ks, sr, mlp_ratio, type, act_layer=nn.ReLU):
        super().__init__()
        pad=ks//2 
        if type=='repdw':
            token_mixer = RepDWConv(dim, ks=ks, stride=1, pad=pad)
            self.block  = nn.Sequential(
                Residual(token_mixer),
                Residual(FFN(dim, dim*mlp_ratio, act_layer)))
            
        elif type=='repsa':
            token_mixer = RepSA(dim, ks=ks, act_layer=act_layer)
            self.block  = nn.Sequential(
                Residual(token_mixer),
                Residual(FFN(dim, dim*mlp_ratio, act_layer)))
            
        elif type=='repmhsa':
            token_mixer = RepMHSA(dim, ks=ks, act_layer=act_layer)
            self.block  = nn.Sequential(
                Residual(token_mixer),
                Residual(FFN(dim, dim*mlp_ratio, act_layer)))
                
        else:
            print("type is not listed")
        
    def forward(self, x):
        return self.block(x)
    
class Stage(nn.Module):
    def __init__(self, dim, depth, cr, ks, sr,  mlp_ratio, type, act_layer=nn.ReLU):
        super().__init__()
        block = [Block(dim=dim, cr=cr, 
                       ks=ks, sr=sr,
                       mlp_ratio=mlp_ratio,
                       type=type, 
                       act_layer=act_layer,
                      )for i in range (depth)]
        self.blocks = nn.Sequential(*block)
    def forward(self, x):
        return self.blocks(x)

class RAViT(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 dims=[ 48, 96, 192, 384],
                 depths=[ 2, 2, 8, 2],
                 ks = [3, 3, 3, 3],
                 cr = [1, 1, 0.5, 0.5],
                 sr = [0, 0, 2, 1],
                 type =["repdw", "repdw", "repdw", "repsa"],
                 ks_pe= 7,
                 patch_size=4, 
                 mlp_ratio=2, 
                 act_layer=nn.GELU, 
                 distillation=False,
                 final_feature_dim=None, 
                 drop_rate=0.0,
                 pretrained=None,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.final_feature_dim = final_feature_dim

        if not isinstance(depths, (list, tuple)):
            depths = [depths] # it means the model has only one stage
        if not isinstance(dims, (list, tuple)):
            dims = [dims]
        
        num_stage = len(depths)
        self.num_stage = num_stage

        stages = []
        patch_embedds=[]
        patch_embedds.append(StemLayer(in_chans, dims[0], ps=patch_size, act_layer=act_layer))

        for i_stage in range(num_stage):
            stage = Stage(
                    dim=dims[i_stage],
                    depth=depths[i_stage], 
                    ks=ks[i_stage],
                    cr=cr[i_stage],
                    sr=sr[i_stage],
                    type=type[i_stage], 
                    mlp_ratio=mlp_ratio, 
                    act_layer=act_layer,
            )
            stages.append(stage)
            if i_stage < (num_stage-1):
                patch_embedd=PatchMerging(dims[i_stage], dims[i_stage+1], ks=ks_pe, act_layer=act_layer)
                patch_embedds.append(patch_embedd)

        self.patch_embedds = nn.Sequential(*patch_embedds)
        self.stages = nn.Sequential(*stages)
        self.head_drop = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

        # # Classifier head
        # if self.final_feature_dim is not None:
        #     if isinstance(self.final_feature_dim, (list, tuple)):
        #         self.pre_head = nn.Sequential(
        #                     Conv2d_BN(dims[-1], self.final_feature_dim[0]),
        #                     nn.AdaptiveAvgPool2d(1)
        #                     )
        #     else:
        #         self.pre_head = nn.AdaptiveAvgPool2d(1)
        #         self.final_feature_dim=[dims[-1], self.final_feature_dim]

        #     self.head = nn.Sequential(
        #         BN_Linear(self.final_feature_dim[0], self.final_feature_dim[1]),
        #         act_layer(),
        #         self.head_drop,
        #         Classfier(self.final_feature_dim[1], num_classes, distillation)
        #         )
        # else:
        #     self.pre_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), self.head_drop)
        #     self.head = Classfier(dims[-1], num_classes, distillation)
        
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
        super(RAViT, self).train(mode)
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
        for i in range(self.num_stage):
            x = self.patch_embedds[i](x)
            x = self.stages[i](x)
            out.append(x)
        return out

    def forward(self, x):
        x = self.forward_feature(x)
        return x

@MODELS.register_module()
class revit_sa22(RAViT):
    def __init__(self, **kwargs):
        super().__init__(
        dims   =[ 48, 96, 192, 384],
        depths =[ 2, 4, 12, 4],
        ks = [3, 3, 7, 7],
        type =["repdw", "repdw", "repsa", "repsa"],
        act_layer=nn.GELU,
        mlp_ratio=3,
        # pretrained='/home/ndr/Container/MMDET_workspace/Backbone/revit_weight/revit_ma_p7_k37_26104.pth.tar'
        )
    def switch_to_deploy(self):
        print("Reparameterize-RAViT Backbone")
        self = reparameterize(self)

@MODELS.register_module()
class ravit_s22(RAViT):
    def __init__(self, **kwargs):
        super().__init__(
        dims   =[ 48, 96, 192, 384],
        depths =[ 2, 4, 12, 4],
        ks = [3, 3, 7, 7],
        type =["repdw", "repdw", "repdw", "repsa"],
        act_layer=nn.GELU,
        mlp_ratio=3,
        pretrained='/home/dsdl/Documents/MMDET_workspace/Backbone/ravit_weight/ravit_s22.pth.tar'
        )
    def switch_to_deploy(self):
        print("Reparameterize-RAViT Backbone")
        self = reparameterize(self)

@MODELS.register_module()
class ravit_s26(RAViT):
    def __init__(self, **kwargs):
        super().__init__(
        dims   =[ 48, 96, 192, 384],
        depths =[ 2, 4, 16, 4],
        ks = [3, 3, 7, 7],
        type =["repdw", "repdw", "repdw", "repsa"],
        act_layer=nn.GELU,
        mlp_ratio=3,
        pretrained='/home/dsdl-4090-ai-server/Documents/MMDET_workspace/Backbone/ravit_weight/ravit_s26.pth.tar'
        )
    def switch_to_deploy(self):
        print("Reparameterize-RAViT Backbone")
        self = reparameterize(self)

@MODELS.register_module()
class ravit_m26(RAViT):
    def __init__(self, **kwargs):
        super().__init__(
        dims   =[ 64, 128, 256, 512],
        depths =[ 2, 4, 16, 4],
        ks = [3, 3, 7, 7],
        type =["repdw", "repdw", "repdw", "repsa"],
        act_layer=nn.GELU,
        mlp_ratio=3,
        pretrained='/home/dsdl-4090-ai-server/Documents/MMDET_workspace/Backbone/ravit_weight/ravit_m26.pth.tar'
        )
    def switch_to_deploy(self):
        print("Reparameterize-RAViT Backbone")
        self = reparameterize(self)

@MODELS.register_module()
class ravit_m22(RAViT):
    def __init__(self, **kwargs):
        super().__init__(
        dims   =[ 64, 128, 256, 512],
        depths =[ 2, 6, 10, 4],
        ks = [3, 3, 7, 7],
        type =["repdw", "repdw", "repdw", "repsa"],
        act_layer=nn.GELU,
        mlp_ratio=3,
        # pretrained='/home/ndr/Container/MMDET_workspace/Backbone/revit_weight/revit_ma_p7_k37_26104.pth.tar'
        )
    def switch_to_deploy(self):
        print("Reparameterize-RAViT Backbone")
        self = reparameterize(self)

def reparameterize(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'reparam'):
            reparametrized = child.reparam()
            setattr(net, child_name, reparametrized)
            reparameterize(reparametrized)
        else:
            reparameterize(child)
    
    return net