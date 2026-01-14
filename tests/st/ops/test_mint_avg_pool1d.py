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
from mindspore import ops, context, mint
from mindspore.device_context.cpu.op_tuning import threads_num


@test_utils.run_with_cell
def avg_pool1d_forward_func(input_x, kernel_size, stride=None, padding=0,
                            ceil_mode=False, count_include_pad=True):
    return mint.nn.functional.avg_pool1d(input_x, kernel_size, stride, padding, ceil_mode, count_include_pad)


@test_utils.run_with_cell
def avg_pool1d_backward_func(input_x, kernel_size, stride=None, padding=0, ceil_mode=False,
                             count_include_pad=True):
    return ops.grad(avg_pool1d_forward_func, (0,))(input_x, kernel_size, stride, padding,
                                                   ceil_mode, count_include_pad)


def set_context(mode):
    if mode == context.GRAPH_MODE:
        context.set_context(mode=mode, jit_config={"jit_level": "O0"})
    else:
        context.set_context(mode=mode)


def compare_result(actual, expected):
    diff = abs(actual.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=20.0)
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_avg_pool1d(mode):
    """
    Feature: Ops
    Description: test op avg_pool1d
    Expectation: expect correct result.
    """
    set_context(mode)
    input_x = Tensor(np.array([[-1.0459, 7.1984, -18.1847, -1.4128, -10.6807],
                               [-3.5300, 14.5506, 3.5729, -1.4509, -18.4536],
                               [3.3068, 10.5899, -1.2154, -8.9516, -2.9720],
                               [-8.6000, 7.2262, 8.4625, -2.2109, -8.7613]]).astype(np.float32))
    out = avg_pool1d_forward_func(input_x, 2, None, 1, False, True)

    print(out)

    expected = np.array([[-0.5229, -5.4931, -6.0468],
                         [-1.7650, 9.0618, -9.9522],
                         [1.6534, 4.6873, -5.9618],
                         [-4.3000, 7.8443, -5.4861]]).astype(np.float32)
    compare_result(out, expected)

    grad = avg_pool1d_backward_func(input_x, 2, None, 1, False, True)

    print("grad out is ", grad)

    expected = np.array([[0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                         [0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                         [0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                         [0.5000, 0.5000, 0.5000, 0.5000, 0.5000]]).astype(np.float32)
    compare_result(grad, expected)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=20.0)
def test_avg_pool1d_dynamic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test op AvgPool1D.
    Expectation: expect AvgPool1D result.
    """
    threads_num(1)
    input_case1 = Tensor(np.random.randn(10, 2, 60), dtype=ms.float32)
    input_case2 = Tensor(np.random.randn(5, 4, 20), dtype=ms.float32)
    TEST_OP(
        avg_pool1d_forward_func,
        [
            [input_case1, 4, 2, 1, False, True],
            [input_case2, 6, 1, 2, True, False],
        ],
        disable_mode=['GRAPH_MODE_GE'],
        disable_case=['ViewTensor', 'EmptyTensor', 'ScalarTensor'],
        case_config={'disable_input_check': True}
    )
