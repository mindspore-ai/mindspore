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

import pytest
import numpy as np
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark
import mindspore as ms
from mindspore import Tensor
from mindspore import context, mint


def set_mode(mode):
    if mode == "GRAPH_MODE":
        context.set_context(mode=context.GRAPH_MODE,
                            jit_config={"jit_level": "O0"})
    else:
        context.set_context(mode=context.PYNATIVE_MODE)


class Net(ms.nn.Cell):

    def __init__(self, size=None, scale_factor=None):
        super(Net, self).__init__()
        self.net = mint.nn.UpsamplingNearest2d(size, scale_factor)

    def construct(self, x):
        return self.net(x)


@test_utils.run_with_cell
def upsample_nearest_forward_func(x, size=None, scale_factor=None):
    net = mint.nn.UpsamplingNearest2d(size, scale_factor)
    return net(x)


@test_utils.run_with_cell
def upsample_nearest_backward_func(x, size=None, scale_factor=None):
    return ms.grad(upsample_nearest_forward_func, (0,))(x, size, scale_factor)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize("mode", ["GRAPH_MODE", "PYNATIVE_MODE"])
def test_upsample_nearest(mode):
    """
    Feature: UpsampleNearest2d
    Description: Test cases for UpsampleNearest2d with output_size.
    Expectation: The result match expected output.
    """
    set_mode(mode)

    input_tensor = Tensor(
        np.array([[[[0.1, 0.3, 0.5], [0.7, 0.9, 1.1]],
                   [[1.3, 1.5, 1.7], [1.9, 2.1, 2.3]]]]).astype(np.float32))
    expected = np.array([[[[0.1000, 0.1000, 0.3000, 0.3000, 0.5000],
                           [0.1000, 0.1000, 0.3000, 0.3000, 0.5000],
                           [0.7000, 0.7000, 0.9000, 0.9000, 1.1000],
                           [0.7000, 0.7000, 0.9000, 0.9000, 1.1000]],
                          [[1.3000, 1.3000, 1.5000, 1.5000, 1.7000],
                           [1.3000, 1.3000, 1.5000, 1.5000, 1.7000],
                           [1.9000, 1.9000, 2.1000, 2.1000, 2.3000],
                           [1.9000, 1.9000, 2.1000, 2.1000,
                            2.3000]]]]).astype(np.float32)
    out = upsample_nearest_forward_func(input_tensor, (4, 5), None)
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)

    expected = np.array([[[[0.1000, 0.1000, 0.3000, 0.3000, 0.5000, 0.5000],
                           [0.1000, 0.1000, 0.3000, 0.3000, 0.5000, 0.5000],
                           [0.7000, 0.7000, 0.9000, 0.9000, 1.1000, 1.1000]],
                          [[1.3000, 1.3000, 1.5000, 1.5000, 1.7000, 1.7000],
                           [1.3000, 1.3000, 1.5000, 1.5000, 1.7000, 1.7000],
                           [1.9000, 1.9000, 2.1000, 2.1000, 2.3000,
                            2.3000]]]]).astype(np.float32)
    out = upsample_nearest_forward_func(input_tensor, None, (1.7, 2.3))
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)

    expected = np.array([[[[4.0, 4.0, 4.0], [2.0, 2.0, 2.0]],
                          [[4.0, 4.0, 4.0], [2.0, 2.0,
                                             2.0]]]]).astype(np.float32)
    out = upsample_nearest_backward_func(input_tensor, None, [1.7, 2.3])
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.01)
def test_upsample_nearest_2d_size_dynamic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test op UpsampleNearest2d and UpsampleNearest2dGrad.
    Expectation: expect UpsampleNearest2d and UpsampleNearest2dGrad result.
    """
    input_case1 = Tensor(np.random.randn(2, 5, 60, 40), dtype=ms.float32)
    input_case2 = Tensor(np.random.randn(4, 3, 15, 30), dtype=ms.float32)
    net = Net((100, 80), None)
    TEST_OP(net, [[input_case1], [input_case2]],
            disable_mode=["GRAPH_MODE_GE"],
            disable_case=['EmptyTensor',
                          'ScalarTensor'],
            case_config={'disable_input_check': True})
