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
import mindspore as ms
import mindspore.ops.operations.math_ops as P
from mindspore import Tensor
from mindspore.common.api import jit


class TraceNet(nn.Cell):

    def __init__(self):
        super(TraceNet, self).__init__()
        self.trace = P.Trace()

    @jit
    def construct(self, x):
        return self.trace(x)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu
@pytest.mark.env_onecard
def test_trace_dyn():
    """
    Feature: test Trace ops in gpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = TraceNet()

    x_dyn = Tensor(shape=[None, None], dtype=ms.float32)
    net.set_inputs(x_dyn)

    x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=ms.float32)
    out = net(x)

    expect_shape = ()
    assert out.asnumpy().shape == expect_shape


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_trace_2d_int32():
    """
    Feature: Returns the sum along diagonals of the int32 array
    Description: 2D x, int32
    Expectation: success
    """
    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")
        x = Tensor(np.array([[1, 20, 5], [4, 5, 9]]).astype(np.int32))
        input_x = x.asnumpy()
        net = TraceNet()
        y = net(x)
        trace_expect = np.trace(input_x).astype(np.int32)
        assert (y.asnumpy() == trace_expect).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_trace_2d_double():
    """
    Feature: Returns the sum along diagonals of the double array
    Description: 2D x, double
    Expectation: success
    """
    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")
        x = Tensor(
            np.array([[3.8, 4.5, 5.], [4.2, 4.5, 6.9]]).astype(np.double))
        input_x = x.asnumpy()
        net = TraceNet()
        y = net(x)
        trace_expect = np.trace(input_x).astype(np.double)
        assert (y.asnumpy() == trace_expect).all()
