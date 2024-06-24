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
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.ops = P.Square()

    def construct(self, x):
        return self.ops(x)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_dynamic_shape_square(dtype):
    """
    Feature: ALL To ALL
    Description: test cases for Square dynamic shape.
    Expectation: the result match to numpy
    """
    input_x_np = np.random.randn(2, 3, 3, 4).astype(dtype)
    benchmark_output = np.square(input_x_np)
    loss = 1e-6
    square_net = Net()
    real_input = Tensor(input_x_np)
    dy_shape = [None for _ in input_x_np.shape]
    input_dyn = Tensor(shape=dy_shape, dtype=real_input.dtype)
    square_net.set_inputs(input_dyn)
    # Graph mode
    context.set_context(mode=context.GRAPH_MODE)
    ms_result = square_net(real_input)
    np.testing.assert_allclose(benchmark_output, ms_result.asnumpy(), rtol=loss, atol=loss)
    # PyNative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    ms_result = square_net(real_input)
    np.testing.assert_allclose(benchmark_output, ms_result.asnumpy(), rtol=loss, atol=loss)
