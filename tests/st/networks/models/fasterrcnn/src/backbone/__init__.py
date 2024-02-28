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
import os
import sys

mindcv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../mindcv")
sys.path.insert(5, mindcv_path)
os.system(f"git submodule init {mindcv_path}")
os.system(f"git submodule update {mindcv_path}")

import ssl
import mindspore as ms
from mindspore import nn, ops
from mindcv.models import create_model
from .fpn import FPN
from .det_resnet import *
from ..utils import logger


def create_backbone(initializer, in_channels=3, pretrained=True, backbone_ckpt=""):
    """
    Creates backbone by MindCV
    Args:
        initializer (str): backbone name.
        in_channels (int): The input channels. Default: 3.
        pretrained (bool): Whether to load the pretrained model. Default: False.
        backbone_ckpt (str): The path of checkpoint files. Default: "".

    """
    if ms.get_auto_parallel_context("device_num") > 1:
        allreduce_sum = ops.AllReduce(ops.ReduceOp.SUM)
        from mindspore.communication.management import get_rank

        local_rank = get_rank() % 8
        # try:
        #     local_rank = get_local_rank()
        # except:
        #     logger.info("Not support get_local_rank, get local_rank by get_rank() % 8")

        if local_rank == 0:
            print(f"==== create_model {local_rank} start")
            net = create_model(
                initializer, in_channels=in_channels, pretrained=pretrained, checkpoint_path=backbone_ckpt
            )
            print(f"==== create_model {local_rank} done")
            r = allreduce_sum(ops.ones((1), ms.float32))
        else:
            r = allreduce_sum(ops.ones((1), ms.float32))
            print(f"==== create_model {local_rank} {r} start")
            net = create_model(
                initializer, in_channels=in_channels, pretrained=pretrained, checkpoint_path=backbone_ckpt
            )
            print(f"==== create_model {local_rank} done")
    else:
        net = create_model(initializer, in_channels=in_channels, pretrained=pretrained, checkpoint_path=backbone_ckpt)
    return net


def build_backbone(cfg):
    model_name = cfg.name
    network = create_backbone(model_name, pretrained=cfg.pretrained)
    if cfg.frozen_bn or cfg.frozen:
        for _, cell in network.cells_and_names():
            if isinstance(cell, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                cell.use_batch_statistics = False
                cell.gamma.requires_grad = False
                cell.beta.requires_grad = False
    if cfg.frozen_2stage:
        network.frozen_2stage = ops.stop_gradient

    if hasattr(cfg, "fpn"):
        network = FPN(
            bottom_up=network,
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            num_outs=cfg.fpn.num_outs,
            level_index=cfg.fpn.level_index,
            norm=cfg.fpn.norm,
            act=cfg.fpn.act,
            upsample_mode=cfg.fpn.upsample_mode,
            frozen=cfg.frozen
        )
    else:
        network = SinOut(network, cfg.in_channels, cfg.out_channel)
    if cfg.frozen:
        for p in network.trainable_params():
            p.requires_grad = False
    return network


class SinOut(nn.Cell):
    def __init__(self, network, in_channels, out_channel):
        super(SinOut, self).__init__()
        self.network = network
        self.out_channel = out_channel
        self.idx = in_channels[-1]
        self.last_conv = nn.Conv2d(in_channels[self.idx], out_channel, kernel_size=1, has_bias=True)

    def construct(self, x):
        x = self.network(x)[self.idx]
        return (self.last_conv(x),)
