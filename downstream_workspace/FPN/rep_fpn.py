# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig, OptConfigType
from timm.models.vision_transformer import trunc_normal_

@MODELS.register_module()
class RepDWConv(nn.Module):
    def __init__(self, dim, ks=5, stride=1, pad=2) -> None:
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
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self,  x: torch.Tensor) -> torch.Tensor:
        xr = self.conv(x) 
        x1, x2, x3, x4 = x.chunk(4, dim=1)
        xr +=torch.cat((self.repconvsxs(x1), 
            self.repconvkx1(x2)+self.repconv1xk(x2),
            self.repconvkx3(x3)+self.repconv3xk(x3),
            self.repconvsxs_s2(x4)), dim=1)
        return self.bn(xr)
    
    def forward_deploy(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
    
    @torch.no_grad()
    def switch_to_deploy(self):
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

@MODELS.register_module()
class Residual(nn.Module):
    def __init__(self, m)-> None:
        super().__init__()
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.m(x)
        
    @torch.no_grad()
    def switch_to_deploy(self):
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

@MODELS.register_module()
class RepMSDWSepConv(nn.Module):
    def __init__(self, inc, ouc, ks=5, stride=1,
                conv_cfg=None, norm_cfg=None, 
                act_cfg=None, inplace=False)-> None:
        super().__init__()
        pad = ks//2
        if stride == 1:
            self.block = nn.Sequential(
                Residual(RepDWConv(inc, ks, 1, pad)),
                ConvModule(inc, ouc, 1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
            )
        else:
            self.block = nn.Sequential(
                RepDWConv(inc, ks, stride, pad),
                ConvModule(inc, ouc, 1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
            )

    def forward(self,  x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

@MODELS.register_module()
class RepFPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Defaults to 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Defaults to -1, which means the
            last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Defaults to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Defaults to False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Defaults to False.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Defaults to None.
        act_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            activation layer in ConvModule. Defaults to None.
        upsample_cfg (:obj:`ConfigDict` or dict, optional): Config dict
            for interpolate layer. Defaults to dict(mode='nearest').
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: Union[bool, str] = False,
        relu_before_extra_convs: bool = False,
        no_norm_on_lateral: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        upsample_cfg: ConfigType = dict(mode='nearest'),
        init_cfg: MultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = RepMSDWSepConv(
                out_channels,
                out_channels,
                ks = 5 if i >= 2 else 3,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = RepMSDWSepConv(
                    in_channels,
                    out_channels,
                    ks = 3 if self.add_extra_convs == 'on_input' else 5,
                    stride=2,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
