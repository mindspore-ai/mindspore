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
Stacked Hourglass Model
"""
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.ops.operations as P

from src.models.layers import Conv, ConvBNReLU, Hourglass, Residual


class StackedHourglassNet(nn.Cell):
    """
    Stacked Hourglass Network
    """

    def __init__(self, nstack, inp_dim, oup_dim):
        super(StackedHourglassNet, self).__init__()

        self.nstack = nstack

        self.input_transpose = P.Transpose()

        self.pre = nn.SequentialCell(
            [
                ConvBNReLU(3, 64, 7, 2),
                Residual(64, 128),
                nn.MaxPool2d(2, 2),
                Residual(128, 128),
                Residual(128, inp_dim),
            ]
        )

        self.hgs = nn.CellList(
            [nn.SequentialCell([Hourglass(4, inp_dim),]) for i in range(nstack)]
        )

        self.features = nn.CellList(
            [nn.SequentialCell([Residual(inp_dim, inp_dim), ConvBNReLU(inp_dim, inp_dim, 1)]) for i in range(nstack)]
        )

        self.outs = nn.CellList([Conv(inp_dim, oup_dim, 1) for i in range(nstack)])
        self.merge_features = nn.CellList([Conv(inp_dim, inp_dim, 1) for i in range(nstack - 1)])
        self.merge_preds = nn.CellList([Conv(oup_dim, inp_dim, 1) for i in range(nstack - 1)])
        self.output_stack = ops.Stack(axis=1)

    def construct(self, imgs):
        """
        forward
        """
        # x size (batch, 3, 256, 256)
        x = self.input_transpose(
            imgs,
            (
                0,
                3,
                1,
                2,
            ),
        )
        x = self.pre(x)
        combined_hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        return self.output_stack(combined_hm_preds)
