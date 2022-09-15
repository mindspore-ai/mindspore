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

import mindspore as ms
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops import operations as P
from scipy import special

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class NetErfc(nn.Cell):
    def __init__(self):
        super(NetErfc, self).__init__()
        self.erfc = P.Erfc()

    def construct(self, x):
        return self.erfc(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_erfc_dshape():
    """
    Feature: Test erfc dynamic shape.
    Description: Test erfc dynamic shape.
    Expectation: Success.
    """
    net = NetErfc()
    input_x_dyn = Tensor(shape=[3, None], dtype=ms.float32)
    net.set_inputs(input_x_dyn)
    input_x = Tensor(np.random.random(([3, 10])), dtype=ms.float32)
    output = net(input_x)
    expect_shape = (3, 10)
    assert output.asnumpy().shape == expect_shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.level0
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('shape', [(2,), (4, 5), (3, 4, 5, 6)])
@pytest.mark.parametrize('dtype, tol', [(np.float16, 1.0e-3), (np.float32, 1.0e-4)])
def test_erfc(mode, shape, dtype, tol):
    """
    Feature: ALL To ALL
    Description: test cases for erf
    Expectation: the result match to scipy
    """
    context.set_context(mode=mode, device_target="CPU")
    erfc = P.Erfc()
    input_x = np.random.randn(*shape).astype(dtype)
    output = erfc(Tensor(input_x))
    expect_output = special.erfc(input_x)
    diff = output.asnumpy() - expect_output
    error = np.ones(shape=expect_output.shape) * tol
    assert np.all(np.abs(diff) < error)
