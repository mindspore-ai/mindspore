# Copyright 2023 Huawei Technologies Co., Ltd
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

from mindspore import Tensor
from mindspore.nn import Cell
from mindspore import ops
import mindspore.context as context
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F

from tests.st.pynative.utils import GradOfAllInputs
from tests.mark_utils import arg_mark


class Fold(Cell):
    def __init__(self, kernel_size, dilation, padding, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def construct(self, input_x, output_size):
        out = ops.fold(input_x, output_size, self.kernel_size, self.dilation,
                       self.padding, self.stride)
        return out


def test_fold_functional_api():
    """
    Feature: test fold functional API.
    Description: test case for fold functional API.
    Expectation: the result match with expected result.
    """
    np.random.seed(1000)
    input_x = Tensor(np.random.randn(1, 4, 4).astype(np.float32))
    output_size = Tensor(np.array([2, 2]).astype(np.int32))
    kernel_size = 1
    dilation = 1
    padding = 0
    stride = 1
    net = Fold(kernel_size, dilation, padding, stride)
    forward_out = net(input_x, output_size)
    expect_forward = np.array([[[[-0.8044583, 0.32093155],
                                 [-0.02548288, 0.6443238]],
                                [[-0.3007967, 0.38947454],
                                 [-0.10743731, -0.47998306]],
                                [[0.5950355, -0.46466753],
                                 [0.6672813, -0.8061156]],
                                [[-1.1960698, -0.40596017],
                                 [-0.18237734, 0.1031929]]]])
    assert np.allclose(expect_forward, forward_out.asnumpy())
    # grad
    out_grad_ms = Tensor(forward_out)
    out_net = Fold(kernel_size, dilation, padding, stride)
    grad_net = GradOfAllInputs(out_net)
    grad_net.set_train()
    grad = grad_net(input_x, output_size, out_grad_ms)
    grad_ms = grad[0].asnumpy()
    expect_grad = np.array([[[-0.8044583, 0.32093155, -0.02548288, 0.6443238],
                             [-0.3007967, 0.38947454, -0.10743731, -0.47998306],
                             [0.5950355, -0.46466753, 0.6672813, -0.8061156],
                             [-1.1960698, -0.40596017, -0.18237734, 0.1031929]]])
    assert np.allclose(expect_grad, grad_ms, atol=1e-4, rtol=1e-4)


def test_fold_tensor_api():
    """
    Feature: test fold tensor API.
    Description: test case for fold tensor API.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.ones([16, 64, 25]), mstype.float32)
    output_size = Tensor([8, 8], mstype.int32)
    output = x.fold(output_size, kernel_size=2,
                    dilation=2, padding=2, stride=2)
    expected_shape = (16, 16, 8, 8)
    assert output.dtype == x.dtype
    assert output.shape == expected_shape


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fold_functional_api_with_invalid_output_size():
    """
    Feature: test fold tensor API with invalid output size.
    Description: test case for fold tensor API.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    x = Tensor(np.ones([16, 64, 4]), mstype.float32)
    output_size = Tensor([6, -1], mstype.int32)
    with pytest.raises(ValueError, match=r"the value of 'output_size' must not be negative"):
        F.fold(x, output_size, kernel_size=2, dilation=2, padding=2, stride=2)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fold_tensor_functional_api_modes():
    """
    Feature: test fold tensor and functional APIs for different modes.
    Description: test case for fold tensor API.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_fold_functional_api()
    test_fold_tensor_api()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    test_fold_functional_api()
    test_fold_tensor_api()


if __name__ == '__main__':
    test_fold_tensor_functional_api_modes()
    test_fold_functional_api_with_invalid_output_size()
