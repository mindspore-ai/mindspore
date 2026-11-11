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
"""Regression coverage for retained fold functions, Tensor methods and gradients."""
import mindspore.common.dtype as mstype
import mindspore.context as context
import numpy as np
import pytest
from mindspore.graph.api import _pynative_executor

from mindspore import Tensor
from mindspore import nn, ops
from mindspore.ops import functional as F
from tests.mark_utils import arg_mark


def test_fold_functional_api():
    """
    Feature: test fold functional API.
    Description: test case for fold functional API.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.ones([16, 64, 25]), mstype.float32)
    output_size = Tensor([8, 8], mstype.int32)
    output = F.fold(x, output_size, kernel_size=[2, 2], dilation=[2, 2], padding=[2, 2], stride=[2, 2])
    expected_shape = (16, 16, 8, 8)
    assert output.dtype == x.dtype
    assert output.shape == expected_shape


def test_fold_tensor_api():
    """
    Feature: test fold tensor API.
    Description: test case for fold tensor API.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.ones([16, 64, 25]), mstype.float32)
    output_size = Tensor([8, 8], mstype.int32)
    output = x.fold(output_size, kernel_size=[2, 2], dilation=[2, 2], padding=[2, 2], stride=[2, 2])
    expected_shape = (16, 16, 8, 8)
    assert output.dtype == x.dtype
    assert output.shape == expected_shape


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_fold_functional_api_with_invalid_output_size():
    """
    Feature: test fold tensor API with invalid output size.
    Description: test case for fold tensor API.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    x = Tensor(np.ones([16, 64, 4]), mstype.float32)
    output_size = Tensor([6, -1], mstype.int32)
    with pytest.raises(ValueError, match=r"the value of 'output_size' must not be negative"):
        F.fold(x, output_size, kernel_size=[2, 2], dilation=[2, 2], padding=[2, 2], stride=[2, 2])
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_fold_tensor_functional_api_modes():
    """
    Feature: test fold tensor and functional APIs for different modes.
    Description: test case for fold tensor API.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    test_fold_functional_api()
    test_fold_tensor_api()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    test_fold_functional_api()
    test_fold_tensor_api()


class FoldRetainedNet(nn.Cell):
    """Exercise the retained Tensor method through the graph parser as well."""
    def __init__(self, tensor_api):
        super().__init__()
        self.tensor_api = tensor_api

    def construct(self, x, output_size):
        if self.tensor_api:
            return x.fold(output_size, kernel_size=(2, 2))
        return ops.fold(x, output_size, kernel_size=(2, 2))


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('tensor_api', [False, True])
def test_fold_retained_overlap_and_gradient(mode, tensor_api):
    """
    Feature: Retained fold APIs sharing the Col2Im implementation.
    Description: Check channel/kernel flattening, overlapping accumulation and its input gradient.
    Expectation: Functional/Tensor fold agree with the reference in both modes.
    """
    context.set_context(mode=mode, device_target="CPU")
    blocks = np.arange(32, dtype=np.float32).reshape(1, 2, 4, 4)
    data = blocks.reshape(1, 8, 4)
    expected = np.zeros((1, 2, 3, 3), dtype=np.float32)
    for row in range(2):
        for col in range(2):
            expected[0, :, row:row + 2, col:col + 2] += blocks[0, :, :, row * 2 + col].reshape(2, 2, 2)
    x, output_size = Tensor(data), Tensor([3, 3], mstype.int32)
    net = FoldRetainedNet(tensor_api)
    np.testing.assert_allclose(net(x, output_size).asnumpy(), expected)
    primitive = ops.Col2Im(kernel_size=(2, 2))
    np.testing.assert_allclose(primitive(x.reshape(1, 2, 4, 4), output_size).asnumpy(), expected)
    np.testing.assert_allclose(ops.grad(net, 0)(x, output_size).asnumpy(), np.ones_like(data))


if __name__ == '__main__':
    test_fold_tensor_functional_api_modes()
    test_fold_functional_api_with_invalid_output_size()
    for execution_mode in (context.GRAPH_MODE, context.PYNATIVE_MODE):
        for use_tensor_api in (False, True):
            test_fold_retained_overlap_and_gradient(execution_mode, use_tensor_api)
