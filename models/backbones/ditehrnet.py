import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule,
                      build_conv_layer, build_norm_layer, constant_init,
                      normal_init)
from torch.nn.modules.batchnorm import _BatchNorm
import torch.utils.checkpoint as cp

from mmpose.utils import get_root_logger
from mmpose.models.builder import BACKBONES
from mmpose.models.backbones.utils import load_checkpoint, channel_shuffle


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                 num_kernels=1):
        super().__init__()

        if num_kernels > 1:
            self.conv = DynamicKernelAggregation(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
                num_kernels=num_kernels)
        else:
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                 num_kernels=1):
        super().__init__()

        if num_kernels > 1:
            self.conv = DynamicKernelAggregation(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
                num_kernels=num_kernels)
        else:
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1BN(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, num_kernels=1):
        super().__init__()

        if num_kernels > 1:
            self.conv = DynamicKernelAggregation(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                groups=groups,
                bias=False,
                num_kernels=num_kernels)
        else:
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                groups=groups,
                bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv1x1BNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, num_kernels=1):
        super().__init__()

        if num_kernels > 1:
            self.conv = DynamicKernelAggregation(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                groups=groups,
                bias=False,
                num_kernels=num_kernels)
        else:
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                groups=groups,
                bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class KernelAttention(nn.Module):
    def __init__(self, channels, reduction=4, num_kernels=4, init_weight=True):
        super().__init__()

        if channels != 3:
            mid_channels = channels // reduction
        else:
            mid_channels = num_kernels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(mid_channels, num_kernels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.avg_pool(x)

        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv2(x).view(x.shape[0], -1)
        x = self.sigmoid(x)

        return x


class KernelAggregation(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, num_kernels,
                 init_weight=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.num_kernels = num_kernels

        self.weight = nn.Parameter(
            torch.randn(num_kernels, out_channels, in_channels // groups, kernel_size, kernel_size),
            requires_grad=True)
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(num_kernels, out_channels))
        else:
            self.bias = None

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.num_kernels):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self, x, attention):
        batch_size, in_channels, height, width = x.size()

        x = x.contiguous().view(1, batch_size * self.in_channels, height, width)

        weight = self.weight.contiguous().view(self.num_kernels, -1)
        weight = torch.mm(attention, weight).contiguous().view(
            batch_size * self.out_channels,
            self.in_channels // self.groups,
            self.kernel_size,
            self.kernel_size)

        if self.bias is not None:
            bias = torch.mm(attention, self.bias).contiguous().view(-1)
            x = F.conv2d(
                x,
                weight=weight,
                bias=bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * batch_size)
        else:
            x = F.conv2d(
                x,
                weight=weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * batch_size)

        x = x.contiguous().view(batch_size, self.out_channels, x.shape[-2], x.shape[-1])

        return x


class DynamicKernelAggregation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_kernels=4):
        super().__init__()
        assert in_channels % groups == 0

        self.attention = KernelAttention(
            in_channels,
            num_kernels=num_kernels)
        self.aggregation = KernelAggregation(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            num_kernels=num_kernels)

    def forward(self, x):
        attention = x

        attention = self.attention(attention)
        x = self.aggregation(x, attention)

        return x


def _split_channels(channels, num_groups):
    split_channels = [channels // num_groups for _ in range(num_groups)]
    split_channels[0] += channels - sum(split_channels)
    return split_channels


class DynamicSplitConvolution(nn.Module):

    def __init__(self, channels, stride, num_branch, num_groups, num_kernels, with_cp=False):
        super().__init__()

        self.with_cp = with_cp

        self.num_groups = num_groups[num_branch]
        self.num_kernels = num_kernels[num_branch]

        self.split_channels = _split_channels(channels, self.num_groups)

        self.conv = nn.ModuleList([
            ConvBN(
                self.split_channels[i],
                self.split_channels[i],
                kernel_size=i * 2 + 3,
                stride=stride,
                padding=i + 1,
                groups=self.split_channels[i],
                num_kernels=self.num_kernels)
            for i in range(self.num_groups)
        ])

    def forward(self, x):

        def _inner_forward(x):
            if self.num_groups == 1:
                x = self.conv[0](x)
            else:
                x_split = torch.split(x, self.split_channels, dim=1)
                x = [conv(t) for conv, t in zip(self.conv, x_split)]
                x = torch.cat(x, dim=1)
                x = channel_shuffle(x, self.num_groups)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class GlobalContextModeling(nn.Module):

    def __init__(self, channels, num_branch, reduction, with_cp=False):
        super().__init__()

        self.with_cp = with_cp

        self.reduction = reduction[num_branch]

        mid_channels = channels // self.reduction

        self.conv_mask = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=2)

        self.channel_attention = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.bn = nn.BatchNorm2d(channels)

    def global_spatial_pool(self, x):
        batch, channel, height, width = x.size()

        # [N, C, H, W]
        x_m = x
        # [N, C, H * W]
        x_m = x_m.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        x_m = x_m.unsqueeze(1)
        # [N, 1, H, W]
        mask = self.conv_mask(x)
        # [N, 1, H * W]
        mask = mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        mask = self.softmax(mask)
        # [N, 1, H * W, 1]
        mask = mask.unsqueeze(-1)
        # [N, 1, C, 1]
        x = torch.matmul(x_m, mask)
        # [N, C, 1, 1]
        x = x.permute(0, 2, 1, 3)

        return x

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            x = self.global_spatial_pool(x)
            x = self.channel_attention(x)
            x = self.bn(identity * x)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class DenseContextModeling(nn.Module):

    def __init__(self, channels, reduction):
        super().__init__()

        num_branches = len(channels)
        self.reduction = reduction[num_branches-2]

        self.channels = channels
        total_channel = sum(channels)
        mid_channels = total_channel // self.reduction

        self.conv_mask = nn.ModuleList([
            nn.Conv2d(channels[i], 1, kernel_size=1, stride=1, padding=0, bias=True)
            for i in range(len(channels))
        ])
        self.softmax = nn.Softmax(dim=2)

        self.channel_attention = nn.Sequential(
            nn.Conv2d(total_channel, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, total_channel, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def global_spatial_pool(self, x, mini_size, i):
        batch, channel, height, width = x.size()
        mini_height, mini_width = mini_size

        # [N, C, H, W]
        x_m = x
        # [N, C, H * W]
        x_m = x_m.view(batch, channel, height * width)
        # [N, MH * MW, C, (H * W) / (MH * MW)]
        x_m = x_m.view(batch, mini_height * mini_width, channel, (height * width) // (mini_height * mini_width))
        # [N, 1, H, W]
        mask = self.conv_mask[i](x)
        # [N, 1, H * W]
        mask = mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        mask = self.softmax(mask)
        # [N, MH * MW, (H * W) / (MH * MW)]
        mask = mask.view(batch, mini_height * mini_width, (height * width) // (mini_height * mini_width))
        # [N, MH * MW, (H * W) / (MH * MW), 1]
        mask = mask.unsqueeze(-1)
        # [N, MH * MW, C, 1]
        x = torch.matmul(x_m, mask)
        # [N, C, MH * MW, 1]
        x = x.permute(0, 2, 1, 3)
        # [N, C, MH, MW]
        x = x.view(batch, channel, mini_height, mini_width)

        return x

    def forward(self, x):
        mini_size = x[-1].size()[-2:]
        out = [self.global_spatial_pool(s, mini_size, i) for s, i in zip(x[:-1], range(len(x)))] + [x[-1]]
        out = torch.cat(out, dim=1)

        out = self.channel_attention(out)

        out = torch.split(out, self.channels, dim=1)
        out = [s * F.interpolate(a, size=s.size()[-2:], mode='nearest') for s, a in zip(x, out)]

        return out


class DynamicMutiScaleContextBlock(nn.Module):

    def __init__(self, in_channels, stride, with_cp=False):
        super().__init__()

        num_branches = len(in_channels)
        branch_channels = [channels // 2 for channels in in_channels]

        self.dcm = DenseContextModeling(
            channels=branch_channels,
            reduction=[8, 8, 8])

        self.conv = nn.ModuleList([
            DynamicSplitConvolution(
                channels=channels,
                stride=stride,
                num_branch=num_branch,
                num_groups=[1, 1, 2, 4],
                num_kernels=[4, 4, 2, 1],
                with_cp=with_cp)
            for channels, num_branch in zip(branch_channels, range(num_branches))
        ])

        self.gcm = nn.ModuleList([
            GlobalContextModeling(
                channels=channels,
                num_branch=num_branch,
                reduction=[4, 4, 8, 8],
                with_cp=with_cp)
            for channels, num_branch in zip(branch_channels, range(num_branches))
        ])

    def forward(self, x):
        x = [s.chunk(2, dim=1) for s in x]
        x1 = [s[0] for s in x]
        x2 = [s[1] for s in x]

        x2 = self.dcm(x2)
        x2 = [conv(s) for s, conv in zip(x2, self.conv)]
        x2 = [p(s) for s, p in zip(x2, self.gcm)]

        x = [torch.cat([s1, s2], dim=1) for s1, s2 in zip(x1, x2)]
        x = [channel_shuffle(s, 2) for s in x]

        return x


class Stem(nn.Module):

    def __init__(self, in_channels, stem_channels, out_channels, with_cp=False):
        super().__init__()

        self.out_channels = out_channels

        self.conv1 = ConvBNReLU(
            in_channels=in_channels,
            out_channels=stem_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            num_kernels=1)

        branch_channels = stem_channels // 2
        mid_channels = branch_channels // 4

        self.branch1 = nn.Sequential(
            ConvBN(
                branch_channels,
                branch_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=branch_channels,
                num_kernels=4),
            GlobalContextModeling(
                branch_channels,
                num_branch=0,
                reduction=[4],
                with_cp=with_cp),
            Conv1x1BNReLU(
                branch_channels,
                branch_channels,
                num_kernels=4)
        )

        self.branch2 = nn.Sequential(
            ConvBN(
                branch_channels,
                branch_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=branch_channels,
                num_kernels=4),
            GlobalContextModeling(
                branch_channels,
                num_branch=0,
                reduction=[4],
                with_cp=with_cp),
            Conv1x1BNReLU(
                branch_channels,
                mid_channels,
                num_kernels=4),
            Conv1x1BN(
                mid_channels,
                branch_channels,
                num_kernels=4),
            ConvBNReLU(
                branch_channels,
                branch_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=branch_channels,
                num_kernels=4)
        )

    def forward(self, x):
        x = self.conv1(x)

        x1, x2 = x.chunk(2, dim=1)

        x1 = self.branch1(x1)
        x2 = self.branch2(x2)

        x = torch.cat((x1, x2), dim=1)
        x = channel_shuffle(x, 2)

        return x


class IterativeHead(nn.Module):

    def __init__(self, in_channels, norm_cfg=dict(type='BN')):
        super().__init__()

        num_branches = len(in_channels)
        self.in_channels = in_channels[::-1]

        projects = []
        for i in range(num_branches):
            if i != num_branches - 1:
                projects.append(
                    DepthwiseSeparableConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=self.in_channels[i + 1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU'),
                        dw_act_cfg=None,
                        pw_act_cfg=dict(type='ReLU')))
            else:
                projects.append(
                    DepthwiseSeparableConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=self.in_channels[i],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU'),
                        dw_act_cfg=None,
                        pw_act_cfg=dict(type='ReLU')))
        self.projects = nn.ModuleList(projects)

    def forward(self, x):
        x = x[::-1]

        y = []
        last_x = None
        for i, s in enumerate(x):
            if last_x is not None:
                last_x = F.interpolate(
                    last_x,
                    size=s.size()[-2:],
                    mode='bilinear',
                    align_corners=True)
                s = s + last_x
            s = self.projects[i](s)
            y.append(s)
            last_x = s

        return y[::-1]


class DiteHRModule(nn.Module):

    def __init__(
            self,
            num_branches,
            num_blocks,
            in_channels,
            multiscale_output=False,
            with_fuse=True,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            with_cp=False,
    ):
        super().__init__()

        self._check_branches(num_branches, in_channels)
        self.in_channels = in_channels
        self.num_branches = num_branches

        self.multiscale_output = multiscale_output
        self.with_fuse = with_fuse
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.with_cp = with_cp

        self.layers = self._make_weighting_blocks(num_blocks)
        if self.with_fuse:
            self.fuse_layers = self._make_fuse_layers()
            self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, in_channels):
        """Check input to avoid ValueError."""
        if num_branches != len(in_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                        f'!= NUM_INCHANNELS({len(in_channels)})'
            raise ValueError(error_msg)

    def _make_weighting_blocks(self, num_blocks, stride=1):
        layers = []
        for i in range(num_blocks):
            layers.append(
                DynamicMutiScaleContextBlock(
                    self.in_channels,
                    stride=stride,
                    with_cp=self.with_cp))

        return nn.Sequential(*layers)

    def _make_fuse_layers(self):
        """Make fuse layer."""
        if self.num_branches == 1:
            return None

        in_channels = self.in_channels
        num_branches = self.num_branches
        num_out_branches = num_branches if self.multiscale_output else 1

        fuse_layers = []
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels[j],
                                in_channels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            build_norm_layer(self.norm_cfg, in_channels[i])[1],
                            nn.Upsample(
                                scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=in_channels[j],
                                        bias=False),
                                    build_norm_layer(self.norm_cfg, in_channels[j])[1],
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[i],
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=False),
                                    build_norm_layer(self.norm_cfg, in_channels[i])[1]))
                        else:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=in_channels[j],
                                        bias=False),
                                    build_norm_layer(self.norm_cfg, in_channels[j])[1],
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=False),
                                    build_norm_layer(self.norm_cfg, in_channels[j])[1],
                                    nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        """Forward function."""
        if self.num_branches == 1:
            return [self.layers[0](x[0])]

        out = self.layers(x)

        if self.with_fuse:
            out_fuse = []
            for i in range(len(self.fuse_layers)):
                y = out[0] if i == 0 else self.fuse_layers[i][0](out[0])
                for j in range(self.num_branches):
                    if i == j:
                        y += out[j]
                    else:
                        y += self.fuse_layers[i][j](out[j])
                out_fuse.append(self.relu(y))
            out = out_fuse
        elif not self.multiscale_output:
            out = [out[0]]

        return out


@BACKBONES.register_module()
class DiteHRNet(nn.Module):
    """Dynamic lightweight High-Resolution Network backbone."""

    def __init__(self,
                 extra,
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 norm_eval=False,
                 with_cp=False):
        super().__init__()

        self.extra = extra
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.stem = Stem(
            in_channels,
            stem_channels=self.extra['stem']['stem_channels'],
            out_channels=self.extra['stem']['out_channels'],
            with_cp=with_cp)

        self.num_stages = self.extra['num_stages']
        self.stages_spec = self.extra['stages_spec']

        num_channels_last = [
            self.stem.out_channels,
        ]
        for i in range(self.num_stages):
            num_channels = self.stages_spec['num_channels'][i]
            num_channels = [num_channels[i] for i in range(len(num_channels))]
            setattr(
                self, 'transition{}'.format(i),
                self._make_transition_layer(num_channels_last, num_channels))

            stage, num_channels_last = self._make_stage(
                self.stages_spec, i, num_channels, multiscale_output=True)
            setattr(self, 'stage{}'.format(i), stage)

        self.with_head = self.extra['with_head']
        if self.with_head:
            self.head_layer = IterativeHead(
                in_channels=num_channels_last,
                norm_cfg=self.norm_cfg)

    def _make_transition_layer(self, num_channels_pre_layer,
                               num_channels_cur_layer):
        """Make transition layer."""
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                num_channels_pre_layer[i],
                                num_channels_pre_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=num_channels_pre_layer[i],
                                bias=False),
                            build_norm_layer(self.norm_cfg, num_channels_pre_layer[i])[1],
                            build_conv_layer(
                                self.conv_cfg,
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            build_norm_layer(self.norm_cfg, num_channels_cur_layer[i])[1],
                            nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] if j == i - num_branches_pre else in_channels
                    conv_downsamples.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels,
                                in_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                groups=in_channels,
                                bias=False),
                            build_norm_layer(self.norm_cfg, in_channels)[1],
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels,
                                out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            build_norm_layer(self.norm_cfg, out_channels)[1],
                            nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv_downsamples))

        return nn.ModuleList(transition_layers)

    def _make_stage(self,
                    stages_spec,
                    stage_index,
                    in_channels,
                    multiscale_output=True):
        num_modules = stages_spec['num_modules'][stage_index]
        num_branches = stages_spec['num_branches'][stage_index]
        num_blocks = stages_spec['num_blocks'][stage_index]
        with_fuse = stages_spec['with_fuse'][stage_index]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            modules.append(
                DiteHRModule(
                    num_branches,
                    num_blocks,
                    in_channels,
                    multiscale_output=reset_multiscale_output,
                    with_fuse=with_fuse,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    with_cp=self.with_cp))
            in_channels = modules[-1].in_channels

        return nn.Sequential(*modules), in_channels

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
                elif isinstance(m, (_BatchNorm, nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        x = self.stem(x)

        y_list = [x]
        for i in range(self.num_stages):
            x_list = []
            transition = getattr(self, 'transition{}'.format(i))
            for j in range(self.stages_spec['num_branches'][i]):
                if transition[j]:
                    if j >= len(y_list):
                        x_list.append(transition[j](y_list[-1]))
                    else:
                        x_list.append(transition[j](y_list[j]))
                else:
                    x_list.append(y_list[j])
            y_list = getattr(self, 'stage{}'.format(i))(x_list)

        x = y_list
        if self.with_head:
            x = self.head_layer(x)

        return [x[0]]

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, (_BatchNorm, nn.BatchNorm2d)):
                    m.eval()
