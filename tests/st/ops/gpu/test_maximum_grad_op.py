# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
import mindspore.context as context
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P
from tests.st.pynative.utils import GradOfAllInputs
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True, device_target="GPU")


class Maximum(Cell):
    def __init__(self):
        super(Maximum, self).__init__()
        self.max = P.Maximum()

    def construct(self, inputa, inputb):
        return self.max(inputa, inputb)


class MaxmumGradNet(Cell):
    def __init__(self):
        super(MaxmumGradNet, self).__init__()
        self.maximum_grad = GradOfAllInputs(Maximum())

    def construct(self, x, y, dy):
        return self.maximum_grad(x, y, dy)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_broadcast_grad_gpu_type():
    """
    Feature: ALL To ALL
    Description: test cases for broadcast_grad of two tensors
    Expectation: the result match to numpy
    """
    np.random.seed(1)
    input_x = np.arange(2 * 3 * 2).reshape((2, 3, 2))
    input_y = np.arange(88, 2 * 3 * 2 + 88).reshape((2, 3, 2))
    input_dout = np.maximum(input_x, input_y)
    net = MaxmumGradNet()
    dtypes = (np.int32, np.int64, np.float16, np.float32, np.float64,
              np.int16, np.uint16, np.uint32, np.uint64)
    for dtype in dtypes:
        result = net(Tensor(input_x.astype(dtype)), Tensor(input_y.astype(dtype)),
                     Tensor(input_dout.astype(dtype)))
        dx = input_dout * (input_x >= input_y)
        dy = input_dout - dx
        assert np.allclose(result[0].asnumpy(), dx, rtol=1.e-4, atol=1.e-8, equal_nan=True)
        assert np.allclose(result[1].asnumpy(), dy, rtol=1.e-4, atol=1.e-8, equal_nan=True)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_max_tensor_grad_with_same_input():
    """
    Feature: test maximumgrad on GPU
    Description: test maximumgrad with same input.
    Expectation: result match to expected result.
    """
    x_np = np.array([0.8, 2.9, 7.2]).astype(np.float32)
    y_np = np.array([0.8, 2.9, 7.2]).astype(np.float32)
    dout = np.array([1.0, -1.0, 0]).astype(np.float32)
    net = MaxmumGradNet()
    output = net(Tensor(x_np), Tensor(y_np), Tensor(dout))
    print(output[0].asnumpy())
    print(output[1].asnumpy())
    expect0 = np.array([0.5, -0.5, 0.])
    expect1 = np.array([0.5, -0.5, 0.])
    assert np.allclose(output[0].asnumpy(), expect0, rtol=1e-6, atol=1e-4)
    assert np.allclose(output[1].asnumpy(), expect1, rtol=1e-6, atol=1e-4)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_max_tensor_grad_with_input_nan():
    """
    Feature: test maximumgrad on GPU
    Description: test maximumgrad with input nan.
    Expectation: result match to expected result.
    """
    x_np = np.array([0.8, 2.9, 7.2]).astype(np.float32)
    y_np = np.full((3,), np.nan).astype(np.float32)
    dout = np.array([1.0, -1.0, 0]).astype(np.float32)
    net = MaxmumGradNet()
    output = net(Tensor(x_np), Tensor(y_np), Tensor(dout))
    print(output[0].asnumpy())
    print(output[1].asnumpy())
    expect0 = np.array([1.0, -1.0, 0])
    expect1 = np.array([1.0, -1.0, 0])
    assert np.allclose(output[0].asnumpy(), expect0, rtol=1e-6, atol=1e-4)
    assert np.allclose(output[1].asnumpy(), expect1, rtol=1e-6, atol=1e-4)
