# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
MindSpore implementation of `ResNet`.
Refer to Deep Residual Learning for Image Recognition.
"""
import os
import sys

mindcv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../mindcv")
sys.path.insert(5, mindcv_path)
os.system(f"git submodule init {mindcv_path}")
os.system(f"git submodule update {mindcv_path}")

from typing import List, Optional, Type, Union

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindcv.models.layers.pooling import GlobalAvgPooling
from mindcv.models.registry import register_model
from mindcv.models.helpers import load_pretrained

from ..utils import logger

__all__ = [
    "ResNet",
    "det_resnet18",
    "det_resnet34",
    "det_resnet50",
    "det_resnet101",
    "det_resnet152",
    "det_resnext50_32x4d",
    "det_resnext101_32x4d",
    "det_resnext101_64x4d",
    "det_resnext152_64x4d",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "conv1",
        "classifier": "classifier",
        **kwargs,
    }


default_cfgs = {
    "det_resnet18": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/resnet/resnet18-1e65cd21.ckpt"),
    "det_resnet34": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/resnet/resnet34-f297d27e.ckpt"),
    "det_resnet50": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/resnet/resnet50-e0733ab8.ckpt"),
    "det_resnet101": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/resnet/resnet101-689c5e77.ckpt"),
    "det_resnet152": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/resnet/resnet152-beb689d8.ckpt"),
}


def get_bn():
    if ms.get_auto_parallel_context("device_num") > 1 and ms.get_context("device_target") == "Ascend":
        return nn.SyncBatchNorm
    return nn.BatchNorm2d


class BasicBlock(nn.Cell):
    """define the basic block of resnet"""
    expansion: int = 1

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 stride: int = 1,
                 groups: int = 1,
                 base_width: int = 64,
                 norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = None) -> None:
        super().__init__()
        if norm is None:
            norm = get_bn()
        assert groups == 1, "BasicBlock only supports groups=1"
        assert base_width == 64, "BasicBlock only supports base_width=64"

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3,
                               stride=stride, padding=1, pad_mode="pad")
        self.bn1 = norm(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               stride=1, padding=1, pad_mode="pad")
        self.bn2 = norm(channels)
        self.down_sample = down_sample

    def construct(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Cell):
    """
    Bottleneck here places the stride for downsampling at 3x3 convolution(self.conv2) as torchvision does,
    while original implementation places the stride at the first 1x1 convolution(self.conv1)
    """
    expansion: int = 4

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 stride: int = 1,
                 groups: int = 1,
                 base_width: int = 64,
                 norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = None) -> None:
        super().__init__()
        if norm is None:
            norm = get_bn()

        width = int(channels * (base_width / 64.0)) * groups

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1)
        self.bn1 = norm(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, pad_mode="pad", group=groups)
        self.bn2 = norm(width)
        self.conv3 = nn.Conv2d(width, channels * self.expansion,
                               kernel_size=1, stride=1)
        self.bn3 = norm(channels * self.expansion)
        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    r"""ResNet model class, based on
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/abs/1512.03385>`_

    Args:
        block: block of resnet.
        layers: number of layers of each stage.
        num_classes: number of classification classes. Default: 1000.
        in_channels: number the channels of the input. Default: 3.
        groups: number of groups for group conv in blocks. Default: 1.
        base_width: base width of pre group hidden channel in blocks. Default: 64.
        norm: normalization layer in blocks. Default: None.
    """

    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 num_classes: int = 1000,
                 in_channels: int = 3,
                 groups: int = 1,
                 base_width: int = 64,
                 norm: Optional[nn.Cell] = None) -> None:
        super().__init__()
        if norm is None:
            norm = get_bn()

        self.norm: nn.Cell = norm  # add type hints to make pylint happy
        self.input_channels = 64
        self.groups = groups
        self.base_with = base_width

        self.conv1 = nn.Conv2d(in_channels, self.input_channels, kernel_size=7,
                               stride=2, pad_mode="pad", padding=3)
        self.bn1 = norm(self.input_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.pool = GlobalAvgPooling()
        self.num_features = 512 * block.expansion
        self.frozen_2stage = ops.identity
        self.expansion = block.expansion

    def _make_layer(self,
                    block: Type[Union[BasicBlock, Bottleneck]],
                    channels: int,
                    block_nums: int,
                    stride: int = 1) -> nn.SequentialCell:
        """build model depending on cfgs"""
        down_sample = None

        if stride != 1 or self.input_channels != channels * block.expansion:
            down_sample = nn.SequentialCell([
                nn.Conv2d(self.input_channels, channels * block.expansion, kernel_size=1, stride=stride),
                self.norm(channels * block.expansion)
            ])

        layers = []
        layers.append(
            block(
                self.input_channels,
                channels,
                stride=stride,
                down_sample=down_sample,
                groups=self.groups,
                base_width=self.base_with,
                norm=self.norm,
            )
        )
        self.input_channels = channels * block.expansion

        for _ in range(1, block_nums):
            layers.append(
                block(
                    self.input_channels,
                    channels,
                    groups=self.groups,
                    base_width=self.base_with,
                    norm=self.norm
                )
            )

        return nn.SequentialCell(layers)

    def construct(self, x):
        """Network forward feature extraction."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        c2 = self.layer1(x)
        c2 = self.frozen_2stage(c2)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5


@register_model
def det_resnet18(pretrained: bool = True, in_channels=3, **kwargs):
    """Get 18 layers ResNet model.
    Refer to the base class `models.ResNet` for more details.
    """
    default_cfg = default_cfgs["det_resnet18"]
    model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=1000, in_channels=in_channels)
        logger.info("load pretrained backbone checkpoint!")

    return model


@register_model
def det_resnet34(pretrained: bool = True, in_channels=3, **kwargs):
    """Get 34 layers ResNet model.
    Refer to the base class `models.ResNet` for more details.
    """
    default_cfg = default_cfgs["det_resnet34"]
    model = ResNet(BasicBlock, [3, 4, 6, 3], in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=1000, in_channels=in_channels)
        logger.info("load pretrained backbone checkpoint!")

    return model


@register_model
def det_resnet50(pretrained: bool = True, in_channels=3, **kwargs):
    """Get 50 layers ResNet model.
    Refer to the base class `models.ResNet` for more details.
    """
    default_cfg = default_cfgs["det_resnet50"]
    model = ResNet(Bottleneck, [3, 4, 6, 3], in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=1000, in_channels=in_channels)
        logger.info("load pretrained backbone checkpoint!")

    return model


@register_model
def det_resnet101(pretrained: bool = True, in_channels=3, **kwargs):
    """Get 101 layers ResNet model.
    Refer to the base class `models.ResNet` for more details.
    """
    default_cfg = default_cfgs["det_resnet101"]
    model = ResNet(Bottleneck, [3, 4, 23, 3], in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=1000, in_channels=in_channels)
        logger.info("load pretrained backbone checkpoint!")

    return model


@register_model
def det_resnet152(pretrained: bool = True, in_channels=3, **kwargs):
    """Get 152 layers ResNet model.
    Refer to the base class `models.ResNet` for more details.
    """
    default_cfg = default_cfgs["det_resnet152"]
    model = ResNet(Bottleneck, [3, 8, 36, 3], in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=1000, in_channels=in_channels)
        logger.info("load pretrained backbone checkpoint!")

    return model


@register_model
def det_resnext50_32x4d(pretrained: bool = True, in_channels=3, **kwargs):
    """Get 50 layers ResNeXt model with 32 groups of GPConv.
    Refer to the base class `models.ResNet` for more details.
    """
    default_cfg = default_cfgs["det_resnext50_32x4d"]
    model = ResNet(Bottleneck, [3, 4, 6, 3], groups=32, base_width=4, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=1000, in_channels=in_channels)
        logger.info("load pretrained backbone checkpoint!")

    return model


@register_model
def det_resnext101_32x4d(pretrained: bool = True, in_channels=3, **kwargs):
    """Get 101 layers ResNeXt model with 32 groups of GPConv.
    Refer to the base class `models.ResNet` for more details.
    """
    default_cfg = default_cfgs["det_resnext101_32x4d"]
    model = ResNet(Bottleneck, [3, 4, 23, 3], groups=32, base_width=4, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=1000, in_channels=in_channels)
        logger.info("load pretrained backbone checkpoint!")

    return model


@register_model
def det_resnext101_64x4d(pretrained: bool = True, in_channels=3, **kwargs):
    """Get 101 layers ResNeXt model with 64 groups of GPConv.
    Refer to the base class `models.ResNet` for more details.
    """
    default_cfg = default_cfgs["det_resnext101_64x4d"]
    model = ResNet(Bottleneck, [3, 4, 23, 3], groups=64, base_width=4, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=1000, in_channels=in_channels)
        logger.info("load pretrained backbone checkpoint!")

    return model


@register_model
def det_resnext152_64x4d(pretrained: bool = True, in_channels=3, **kwargs):
    default_cfg = default_cfgs["det_resnext101_64x4d"]
    model = ResNet(Bottleneck, [3, 8, 36, 3], groups=64, base_width=4, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=1000, in_channels=in_channels)
        logger.info("load pretrained backbone checkpoint!")

    return model
