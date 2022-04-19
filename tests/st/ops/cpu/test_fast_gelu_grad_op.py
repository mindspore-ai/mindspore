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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import composite as C
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class FastGeluNet(nn.Cell):
    """FastGeluNet."""

    def __init__(self):
        """Init."""
        super(FastGeluNet, self).__init__()
        self.fast_gelu = P.FastGeLU()

    def construct(self, x):
        """Construct."""
        return self.fast_gelu(x)


class FastGeLUGrad(nn.Cell):
    """FastGeLUGrad."""

    def __init__(self, network):
        """Init."""
        super(FastGeLUGrad, self).__init__()
        self.fast_gelu_grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_data, sens):
        """Construct."""
        gout = self.fast_gelu_grad(self.network)(input_data, sens)
        return gout


def np_all_close_with_loss(out, expect):
    """np_all_close_with_loss"""
    return np.allclose(out, expect, 0.005, 0.005, equal_nan=True)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fast_gelu_grad_float32():
    """
    Feature: FastGeLUGrad cpu kernel
    Description: test the rightness of FastGeLUGrad cpu kernel, type of input data is float32.
    Expectation: Success.
    """
    x_ms = Tensor(np.array([0.58401114, 0.68800163, 0.9760397, 0.14702141, 0.46563736, 0.9607501,
                            0.14567593, 0.12261796, 0.37054458, 0.46421242]).astype(np.float32))
    dy_ms = Tensor(np.array([0.5559598, 0.96994054, 0.24770357, 0.34646875, 0.2984393, 0.03287048,
                             0.55681044, 0.966908, 0.06015943, 0.6099489]).astype(np.float32))

    net = FastGeluNet()
    grad = FastGeLUGrad(net)

    output = grad(x_ms, dy_ms)
    expect = [0.51520324, 0.9445478, 0.26364434, 0.21633989, 0.2560539, 0.03488608,
              0.34668517, 0.5836204, 0.04784583, 0.52357304]
    assert np_all_close_with_loss(output[0].asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_fast_gelu_grad_float16():
    """
    Feature: FastGeLUGrad cpu kernel
    Description: test the rightness of FastGeLUGrad cpu kernel, type of input data is float16.
    Expectation: Success.
    """
    x_ms = Tensor(np.array([0.58401114, 0.68800163, 0.9760397, 0.14702141, 0.46563736, 0.9607501,
                            0.14567593, 0.12261796, 0.37054458, 0.46421242]).astype(np.float16))
    dy_ms = Tensor(np.array([0.5559598, 0.96994054, 0.24770357, 0.34646875, 0.2984393, 0.03287048,
                             0.55681044, 0.966908, 0.06015943, 0.6099489]).astype(np.float16))

    net = FastGeluNet()
    grad = FastGeLUGrad(net)

    output = grad(x_ms, dy_ms)
    expect = [0.5156, 0.9443, 0.2637, 0.2157, 0.2559, 0.03488, 0.3464, 0.5835, 0.04785, 0.5234]
    assert np_all_close_with_loss(output[0].asnumpy(), expect)
