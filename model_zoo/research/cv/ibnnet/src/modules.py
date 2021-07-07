# Copyright 2021 Huawei Technologies Co., Ltd
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
python modules.py
"""
import mindspore.nn as nn
import mindspore.ops as ops


class IBN(nn.Cell):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`

    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """

    def __init__(self, planes, ratio=0.5):
        super(IBN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.GroupNorm(self.half, self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def construct(self, x):
        op_split = ops.Split(1, 2)
        split = op_split(x)
        out1 = self.IN(split[0])
        out2 = self.BN(split[1])
        op_cat = ops.Concat(1)
        out = op_cat((out1, out2))
        return out


class SELayer(nn.Cell):
    """SELayer

    Args:
        x (Tensor): input tensor
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = ops.ReduceMean()
        self.fc = nn.SequentialCell(
            [
                nn.Dense(channel, int(channel / reduction), has_bias=False),
                nn.ReLU(),
                nn.Dense(int(channel / reduction), channel, has_bias=False),
                nn.Sigmoid()
            ]
        )

    def construct(self, x):
        [b, c, _, _] = x.shape
        _reshape = ops.Reshape()
        y = _reshape(self.avg_pool(x, (2, 3)), (b, c))
        y = _reshape(self.fc(y), (b, c, 1, 1))
        broadcast = ops.BroadcastTo(x.shape)
        return x * broadcast(y)
