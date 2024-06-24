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
from tests.mark_utils import arg_mark

import pytest
import numpy as np
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
import mindspore as ms
from mindspore import Tensor
from mindspore import ops, context, mint


@test_utils.run_with_cell
def fold_forward_func(input_tensor, output_size, kernel_size, dilation=1, padding=0, stride=1):
    return mint.nn.functional.fold(input_tensor, output_size, kernel_size, dilation, padding, stride)


@test_utils.run_with_cell
def fold_backward_func(input_tensor, output_size, kernel_size, dilation=1, padding=0, stride=1):
    return ops.grad(fold_forward_func, (0,))(input_tensor, output_size, kernel_size, dilation, padding, stride)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fold(mode):
    """
    Feature: test ops.
    Description: test op Col2ImExt.
    Expectation: success.
    """
    context.set_context(mode=mode)
    input_tensor = Tensor(np.array([[0.1683, -0.1127, 0.4832, 2.7538],
                                    [0.7056, -0.2466, -0.4011, -1.4433],
                                    [-0.2314, 0.0880, -0.7491, -0.1695],
                                    [0.1973, -0.2653, 0.9415, 1.1586]]).astype(np.float32))
    expected = np.array([[[0.1973, 0.0000, 0.0880, -0.2653],
                          [0.0000, 0.0000, 0.0000, 0.0000],
                          [-0.4011, 0.0000, 2.7538, -1.4433],
                          [0.9415, 0.0000, -0.1695, 1.1586]]]).astype(np.float32)
    out = fold_forward_func(input_tensor, (4, 4), 2, 1, 1, 3)
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)

    expected = np.array([[0., 0., 0., 1.],
                         [0., 0., 1., 1.],
                         [0., 1., 0., 1.],
                         [1., 1., 1., 1.]]).astype(np.float32)
    grad = fold_backward_func(input_tensor, (4, 4), 2, 1, 1, 3)
    diff = abs(grad.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fold_dynamic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test op Col2ImExt and Im2ColExt.
    Expectation: expect correct result.
    """
    ms.context.set_context(
        runtime_num_threads=1
    )  # multi-threads have none-initialized bug now.
    input_case1 = Tensor(np.random.randn(8, 30, 3), dtype=ms.float32)
    input_case2 = Tensor(np.random.randn(9, 4), dtype=ms.float32)
    TEST_OP(
        fold_forward_func,
        [
            [input_case1, (12, 16), (6, 5), (3, 2), (2, 3), (2, 5)],
            [input_case2, (8, 8), (3, 3), (2, 2), (1, 1), (3, 3)],
        ],
        "col2im_ext"
    )
