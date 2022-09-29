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
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import composite as C
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class Net(nn.Cell):
    def __init__(self, sample, replacement, seed=0):
        super(Net, self).__init__()
        self.sample = sample
        self.replacement = replacement
        self.seed = seed

    def construct(self, x):
        return C.multinomial(x, self.sample, self.replacement, self.seed)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_multinomial_net():
    x0 = Tensor(np.array([0.9, 0.2]).astype(np.float32))
    x1 = Tensor(np.array([[0.9, 0.2], [0.9, 0.2]]).astype(np.float32))
    net0 = Net(1, True, 20)
    net1 = Net(2, True, 20)
    net2 = Net(6, True, 20)
    out0 = net0(x0)
    out1 = net1(x0)
    out2 = net2(x1)
    assert out0.asnumpy().shape == (1,)
    assert out1.asnumpy().shape == (2,)
    assert out2.asnumpy().shape == (2, 6)


class DynamicShapeNet(nn.Cell):
    """
    Inputs:
        - **x** (Tensor) - the input tensor containing the cumsum of probabilities, must be 1 or 2
          dimensions. Must be one of the following types: float16, float32, float64. CPU and GPU
          supports x 1 or 2 dimensions and Ascend only supports 2 dimensions.
        - **num_samples** (int) - number of samples to draw, must be a nonnegative number.
    """
    def __init__(self):
        super(DynamicShapeNet, self).__init__()
        self.unique = P.Unique()
        self.gather = P.Gather()
        self.multinomial = P.Multinomial()

    def construct(self, x, indices):
        unique_indices, _ = self.unique(indices)
        x = self.gather(x, unique_indices, 0)
        return self.multinomial(x, 2)


@pytest.mark.level2
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_multinomial_dynamic_shape():
    """
    Feature: test Multinomial dynamic_shape feature.
    Description: test Multinomial dynamic_shape feature. Only support GRAPH_MODE.
    Expectation: success.
    """
    # dynamic inputs
    indices_np = np.random.randint(0, 3, size=6)
    indices_ms = Tensor(indices_np)

    # data preparation
    x = Tensor(np.arange(20).reshape(4, 5).astype(np.float32) / 10)

    # dynamic shape
    x_dyn = Tensor(shape=[None for _ in x.shape], dtype=x.dtype)
    dynamic_shape_net = DynamicShapeNet()
    dynamic_shape_net.set_inputs(x_dyn, indices_ms)

    # run in graph mode
    outputs = dynamic_shape_net(x, indices_ms)
    expect_shape = (len(np.unique(indices_np)), 2)
    assert outputs.asnumpy().shape == expect_shape
