# Copyright 2022 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class OriginNet(nn.Cell):
    def __init__(self):
        super(OriginNet, self).__init__()
        self.matmul = P.MatMul()
        self.bias_add = P.BiasAdd()
        self.relu = P.ReLU()

    def construct(self, x, y, b):
        matmul = self.matmul(x, y)
        bias_add = self.bias_add(matmul, b)
        relu = self.relu(bias_add)
        return relu


def numpy_func(x, y, b):
    matmul = np.matmul(x, y)
    bias_add = np.add(matmul, b)
    relu = np.maximum(0, bias_add)
    return relu


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_matmul_biasadd_fusion():
    """
    Feature: MatmulBiasadd Fusion test
    Description: The output is correct after fusion
    Expectation: success
    """
    x_np = np.arange(1 * 3).reshape((1, 3)).astype(np.float32)
    y_np = np.arange(3 * 5).reshape((3, 5)).astype(np.float32)
    b_np = np.array([0.1, 0.3, 0.5, 0.7, 0.9]).astype(np.float32)
    x = Tensor(x_np, ms.float32)
    y = Tensor(y_np, ms.float32)
    b = Tensor(b_np, ms.float32)
    net = OriginNet()
    output = net(x, y, b)
    expect = numpy_func(x_np, y_np, b_np)
    assert np.allclose(output.asnumpy(), expect)
