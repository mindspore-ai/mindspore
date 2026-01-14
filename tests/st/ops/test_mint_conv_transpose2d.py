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
import pytest
import numpy as np
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark

import mindspore as ms
from mindspore import Tensor, context
from mindspore import mint
from mindspore.device_context.cpu.op_tuning import threads_num


@test_utils.run_with_cell
def conv_transpose2d_forward_func(x, weight, bias, stride=1, padding=0,
                                  output_padding=0, groups=1, dilation=1):
    return mint.nn.functional.conv_transpose2d(
        x, weight, bias, stride, padding, output_padding, groups, dilation
    )


@test_utils.run_with_cell
def conv_transpose2d_backward_func(x, weight, bias, stride=1, padding=0,
                                   output_padding=0, groups=1, dilation=1):
    return ms.grad(conv_transpose2d_forward_func, (0, 1, 2))(
        x, weight, bias, stride, padding, output_padding, groups, dilation
    )

def set_mode(mode):
    """
    set mode
    """
    if mode == "KBK":
        context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    else:
        context.set_context(mode=context.PYNATIVE_MODE)


@pytest.mark.parametrize("mode", ["KBK", "PYBOOST"])
def test_conv_transpose2d_hf32(mode):
    """
    Feature: ConvTranspose2D
    Description: test hf32 with mint.nn.functional.conv_transpose2d
    Expectation: expect correct result.
    """
    set_mode(mode)
    x = np.array([[4.17021990e-1, 7.20324516e-1],
                  [1.14374816e-4, 3.02332580e-1]]).astype(np.float32)
    x = Tensor(np.reshape(x, (1, 1, 2, 2)))
    w = np.array([[0.14675589, 0.09233859, 0.18626021, 0.34556073],
                  [0.39676747, 0.53881675, 0.41919452, 0.6852195]]).astype(np.float32)
    w = Tensor(np.reshape(w, (1, 2, 2, 2)))
    b = Tensor(np.array([0.20445225, 0.87811744]).astype(np.float32))

    expect_out = np.array([[0.2656369, 0.34863594, 0.27096134],
                           [0.28214604, 0.52709454, 0.4812674],
                           [0.20447356, 0.2608167, 0.30894494],
                           [1.0435501, 1.3885303, 1.2661824],
                           [1.0529616, 1.5858095, 1.5346042],
                           [0.87816536, 1.0049454, 1.0853312]]).astype(np.float32)
    expect_out = np.reshape(expect_out, (1, 2, 3, 3))
    expect_dx = np.array([2.8109741] * 4).astype(np.float32)
    expect_dx = np.reshape(expect_dx, (1, 1, 2, 2))
    expect_dw = np.array([1.4396896] * 8).astype(np.float32)
    expect_dw = np.reshape(expect_dw, (1, 2, 2, 2))
    expect_db = np.array([9., 9.]).astype(np.float32)

    out = conv_transpose2d_forward_func(x, w, b)
    dx, dw, db = conv_transpose2d_backward_func(x, w, b)
    loss = 1e-5
    assert np.allclose(out.asnumpy(), expect_out, loss, loss)
    assert np.allclose(dx.asnumpy(), expect_dx, loss, loss)
    assert np.allclose(dw.asnumpy(), expect_dw, loss, loss)
    assert np.allclose(db.asnumpy(), expect_db, loss, loss)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=4.01
)
def test_conv_transpose2d_dynamic():
    """
    Feature: Ops
    Description: test op conv_transpose2d dynamic shape
    Expectation: expect correct result.
    """
    threads_num(1)
    # test case 1
    x = Tensor(np.random.randn(1, 4, 5, 5), dtype=ms.float32)
    w = Tensor(np.random.randn(4, 8, 3, 3), dtype=ms.float32)
    b = Tensor(np.random.randn(8), dtype=ms.float32)
    stride = (1, 1)
    padding = (0, 0)
    output_padding = (0, 0)
    groups = 1
    dilation = (1, 1)
    input_case1 = [x, w, b, stride, padding, output_padding, groups, dilation]
    # test case 2
    x = Tensor(np.random.randn(10, 20, 15), dtype=ms.float32)
    w = Tensor(np.random.randn(10, 12, 6, 6), dtype=ms.float32)
    b = Tensor(np.random.randn(24), dtype=ms.float32)
    stride = (4, 3)
    padding = (2, 3)
    output_padding = (1, 0)
    groups = 2
    dilation = (1, 2)
    input_case2 = [x, w, b, stride, padding, output_padding, groups, dilation]
    TEST_OP(
        conv_transpose2d_forward_func,
        [input_case1, input_case2],
        disable_mode=["GRAPH_MODE_GE"],
        disable_case=['EmptyTensor',
                      'ScalarTensor'],
        case_config={'disable_input_check': True},
    )
