# Copyright 2020 Huawei Technologies Co., Ltd
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

import mindspore
import mindspore.nn as nn
import mindspore.context as context

from mindspore import Tensor
from mindspore.ops.composite import GradOperation


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pad_basic():
    # confirm array is being padded with 0's
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    test_arr = np.array([[1, 2], [3, 4]]).astype(np.float32)
    test_arr_expected = np.array(
        [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]]).astype(np.float32)
    x_test = Tensor(test_arr, dtype=mindspore.float32)

    pad_op = nn.Pad(mode='CONSTANT', paddings=((1, 1), (1, 1)))
    y_test = pad_op(x_test).asnumpy()

    np.testing.assert_array_equal(y_test, test_arr_expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pad_row():
    # Confirm correct row padding
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    test_arr_1 = np.random.rand(40, 40).astype(np.float32)
    test_paddings_1 = ((2, 3), (0, 0))

    test_arr_2 = np.random.randn(3, 10, 30, 30).astype(np.float32)
    test_paddings_2 = ((0, 0), (0, 0), (3, 0), (0, 0))

    pad_op_row_1 = nn.Pad(mode='CONSTANT', paddings=test_paddings_1)
    pad_op_row_2 = nn.Pad(mode='CONSTANT', paddings=test_paddings_2)

    x_test_1 = Tensor(np.array(test_arr_1), dtype=mindspore.float32)
    x_test_2 = Tensor(np.array(test_arr_2), dtype=mindspore.float32)

    y_test_1 = pad_op_row_1(x_test_1).asnumpy()
    y_test_2 = pad_op_row_2(x_test_2).asnumpy()

    # check size
    assert y_test_1.shape == (45, 40)
    assert y_test_2.shape == (3, 10, 33, 30)

    # check values - select correct sections
    np.testing.assert_equal(y_test_1[2:-3, :], test_arr_1)
    np.testing.assert_equal(y_test_2[:, :, 3:, :], test_arr_2)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pad_column():
    # Confirm correct column padding
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    test_arr_1 = np.random.randn(40, 40).astype(np.float32)
    test_paddings_1 = ((0, 0), (3, 3))

    test_arr_2 = np.random.randn(3, 10, 30, 30).astype(np.float32)
    test_paddings_2 = ((0, 0), (0, 0), (0, 0), (6, 1))

    pad_op_col_1 = nn.Pad(mode='CONSTANT', paddings=test_paddings_1)
    pad_op_col_2 = nn.Pad(mode='CONSTANT', paddings=test_paddings_2)

    x_test_1 = Tensor(np.array(test_arr_1), dtype=mindspore.float32)
    x_test_2 = Tensor(np.array(test_arr_2), dtype=mindspore.float32)

    y_test_1 = pad_op_col_1(x_test_1).asnumpy()
    y_test_2 = pad_op_col_2(x_test_2).asnumpy()

    # check size
    assert y_test_1.shape == (40, 46)
    assert y_test_2.shape == (3, 10, 30, 37)

    # check values - select correct sections - should match
    np.testing.assert_equal(y_test_1[:, 3:-3], test_arr_1)
    np.testing.assert_equal(y_test_2[:, :, :, 6:-1], test_arr_2)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pad_3d_pad():
    # Confirm correct 3d padding - row, column, channel
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    test_arr = np.random.randn(5, 3, 30, 30).astype(np.float32)
    test_paddings = ((0, 0), (2, 1), (0, 1), (0, 2))  # padding 3 dims now

    pad_op_3d = nn.Pad(mode='CONSTANT', paddings=test_paddings)
    x_test = Tensor(np.array(test_arr), dtype=mindspore.float32)

    y_test = pad_op_3d(x_test).asnumpy()
    assert y_test.shape == (5, 6, 31, 32)
    np.testing.assert_equal(test_arr, y_test[:, 2:-1, :-1, :-2])


# For testing backprop
class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_, output_grad):
        return self.grad(self.network)(input_, output_grad)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.pad = nn.Pad(mode="CONSTANT", paddings=(
            (0, 0), (4, 3), (1, 1), (0, 2)))

    def construct(self, x):
        return self.pad(x)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pad_3d_backprop():
    # Confirm correct 3d padding backprop
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    test_arr = np.random.randn(5, 3, 30, 30).astype(np.float32)
    x_test = Tensor(test_arr, dtype=mindspore.float32)

    padded_shape = (5, 10, 32, 32)
    dy = np.random.randn(*padded_shape).astype(np.float32)
    expected_dx = dy[:, 4:-3, 1:-1, :-2]

    net = Grad(Net())
    dx = net(x_test, Tensor(dy))
    dx = dx[0].asnumpy()
    np.testing.assert_array_equal(dx, expected_dx)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pad_error_cases():
    # Test against common errorneous inputs to catch correctly
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    # TEST 1 - Neg padding values
    test_op = nn.Pad(paddings=((0, 0), (-1, -1)), mode="CONSTANT")
    test_arr = np.random.randn(3, 3)
    test_arr_ms = Tensor(test_arr, dtype=mindspore.float32)

    with pytest.raises(ValueError):
        test_op(test_arr_ms)

    # TEST 2 - Mismatched input size and paddings - 1D tensor
    test_op = nn.Pad(paddings=((0, 0), (1, 0)), mode="CONSTANT")
    test_arr = np.random.randn(3)  # 1D Tensor
    test_arr_ms = Tensor(test_arr, dtype=mindspore.float32)

    with pytest.raises(ValueError):
        test_op(test_arr_ms)

    # TEST 3 - Mismatched input size and paddings - 2D tensor, 3D padding
    test_op = nn.Pad(paddings=((0, 0), (1, 0)), mode="CONSTANT")  # 2D Padding
    test_arr = np.random.randn(1, 3, 3)  # 3D Tensor
    test_arr_ms = Tensor(test_arr, dtype=mindspore.float32)

    with pytest.raises(ValueError):
        test_op(test_arr_ms)

    # TEST 4 - 1D Paddings should not work
    with pytest.raises(TypeError):
        test_op = nn.Pad(paddings=((0, 2)), mode="CONSTANT")

    # TEST 5 - Padding beyond 4d - (added check in nn file in PR)
    with pytest.raises(ValueError):
        _ = nn.Pad(paddings=((0, 0), (0, 0,), (0, 0), (0, 0),
                             (1, 0)), mode="CONSTANT")  # 2D Padding
