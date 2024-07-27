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
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
import mindspore as ms
from mindspore import Tensor
from mindspore import ops, context, mint


@test_utils.run_with_cell
def avg_pool2d_forward_func(image, kernel_size, stride=None, padding=0,
                            ceil_mode=False, count_include_pad=True, divisor_override=None,):
    return mint.nn.functional.avg_pool2d(image, kernel_size, stride, padding,
                                         ceil_mode, count_include_pad, divisor_override,)


@test_utils.run_with_cell
def avg_pool2d_backward_func(image, kernel_size, stride=None, padding=0, ceil_mode=False,
                             count_include_pad=True, divisor_override=None):
    return ops.grad(avg_pool2d_forward_func, (0,))(image, kernel_size, stride, padding,
                                                   ceil_mode, count_include_pad, divisor_override)


@test_utils.run_with_cell
def avg_pool2d_backward_forward_func(grad, image, kernel_size, stride, padding=0, ceil_mode=False,
                                     count_include_pad=True, divisor_override=None,):
    return ops.auto_generate.AvgPool2DGrad()(grad, image, kernel_size, stride, padding, ceil_mode,
                                             count_include_pad, divisor_override)


@test_utils.run_with_cell
def avg_pool2d_double_backward_func(grad, image, kernel_size, stride, padding=0, ceil_mode=False,
                                    count_include_pad=True, divisor_override=None,):
    return ops.grad(avg_pool2d_backward_forward_func, (0))(grad, image, kernel_size, stride, padding,
                                                           ceil_mode, count_include_pad, divisor_override,)


def set_context(mode):
    if mode == context.GRAPH_MODE:
        context.set_context(mode=mode, jit_config={"jit_level": "O0"})
    else:
        context.set_context(mode=mode)


def compare_result(actual, expected):
    diff = abs(actual.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)


@pytest.mark.level3
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_avg_pool2d(mode):
    """
    Feature: Ops
    Description: test op avg_pool2d and avg_pool2d_grad
    Expectation: expect correct result.
    """
    set_context(mode)
    image = Tensor(np.array([[[4.1702e-1, 7.2032e-1, 1.1437e-4, 3.0223e-1],
                              [1.4676e-1, 9.2339e-2, 1.8626e-1, 3.4556e-1],
                              [3.9677e-1, 5.3882e-1, 4.1919e-1, 6.8522e-1],
                              [2.0445e-1, 8.7812e-1, 2.7338e-2, 6.7047e-1]]]).astype(np.float32))
    out = avg_pool2d_forward_func(image, 2, None, 1, False, True)

    expected = np.array([[[0.1043, 0.1801, 0.0756],
                          [0.1359, 0.3092, 0.2577],
                          [0.0511, 0.2264, 0.1676]]]).astype(np.float32)
    compare_result(out, expected)

    grad = avg_pool2d_backward_func(image, 2, 2, 0, False, False)

    expected = np.array([[[0.2500, 0.2500, 0.2500, 0.2500],
                          [0.2500, 0.2500, 0.2500, 0.2500],
                          [0.2500, 0.2500, 0.2500, 0.2500],
                          [0.2500, 0.2500, 0.2500, 0.2500]]]).astype(np.float32)
    compare_result(grad, expected)


@pytest.mark.level3
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.parametrize("mode", [context.GRAPH_MODE])
def test_avg_pool2d_double_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op SoftmaxBackward.
    Expectation: expect correct result.
    """
    set_context(mode)
    image = Tensor(np.random.rand(4, 3, 10, 10).astype(np.float32))
    grad = avg_pool2d_forward_func(image, 4)
    double_grad = avg_pool2d_double_backward_func(grad, image, 4, 2)
    expected = np.ones((4, 3, 4, 4)).astype(np.float32)
    compare_result(double_grad, expected)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_avg_pool2d_dynamic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test op AvgPool2D and AvgPool2DGrad.
    Expectation: expect AvgPool2D and AvgPool2DGrad. result.
    """
    context.set_context(
        runtime_num_threads=1
    )  # multi-threads have none-initialized bug now.
    input_case1 = Tensor(np.random.randn(10, 2, 5, 60), dtype=ms.float32)
    input_case2 = Tensor(np.random.randn(5, 4, 20, 15), dtype=ms.float32)
    TEST_OP(
        avg_pool2d_forward_func,
        [
            [input_case1, 4, (2, 2), (1,), False, True, 1],
            [input_case2, 6, (1, 1), (2,), True, False, 2],
        ],
        'avg_pool2d',
        disable_mode=["GRAPH_MODE"]
    )
