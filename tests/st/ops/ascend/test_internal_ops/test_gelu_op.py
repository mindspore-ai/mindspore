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
import numpy as np
import pytest

import mindspore as ms
from mindspore import context, nn, Tensor, ops


class GeluNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.gelu = ops.GeLU()

    def construct(self, x):
        return self.gelu(x)


def gelu_net(x_shape, dtype):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    x = np.random.randn(*x_shape)

    net = GeluNet()
    _ = net(Tensor(x, dtype=dtype))


def test_gelu():
    """
    Feature: test gelu operator in graph mode.
    Description: test gelu.
    Expectation: the result is correct
    """
    x_shape = (1,)
    dtype = ms.bfloat16
    gelu_net(x_shape, dtype)
