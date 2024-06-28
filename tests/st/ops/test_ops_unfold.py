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
def unfold_forward_func(input_tensor, kernel_size, dilation=1, padding=0, stride=1):
    return mint.nn.functional.unfold(input_tensor, kernel_size, dilation, padding, stride)


@test_utils.run_with_cell
def unfold_backward_func(input_tensor, kernel_size, dilation=1, padding=0, stride=1):
    return ops.grad(unfold_forward_func, (0,))(input_tensor, kernel_size, dilation, padding, stride)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_unfold(mode):
    """
    Feature: test ops.
    Description: test op Im2ColExt.
    Expectation: success.
    """
    context.set_context(mode=mode)
    input_tensor = Tensor(np.array([[[[0.6353, 0.5796, -1.6168, -0.4002],
                                      [0.9113, 0.0809, -1.1446, -1.0034],
                                      [0.5508, 1.5306, -0.4610, -1.6681],
                                      [0.6266, 1.0720, -1.4088, -0.2456]]]]).astype(np.float32))
    expected = np.array([[[[0.0000, 0.0000, 0.0000, -0.4610],
                           [0.0000, 0.0000, 0.5508, -1.6681],
                           [0.0000, -1.6168, 0.0000, -1.4088],
                           [0.6353, -0.4002, 0.6266, -0.2456]]]]).astype(np.float32)
    out = unfold_forward_func(input_tensor, 2, 1, 1, 3)
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)

    expected = np.array([[[[1., 0., 1., 1.],
                           [0., 0., 0., 0.],
                           [1., 0., 1., 1.],
                           [1., 0., 1., 1.]]]]).astype(np.float32)
    grad = unfold_backward_func(input_tensor, 2, 1, 1, 3)
    diff = abs(grad.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_unfold_dynamic(mode):
    """
    Feature: test dynamic by TEST_OP.
    Description: test op Im2ColExt and Col2ImExt.
    Expectation: expect correct result.
    """
    ms.context.set_context(
        runtime_num_threads=1
    )  # multi-threads have none-initialized bug now.
    input_case1 = Tensor(np.random.randn(2, 5, 60, 30), dtype=ms.float32)
    input_case2 = Tensor(np.random.randn(1, 3, 15, 10), dtype=ms.float32)
    TEST_OP(
        unfold_forward_func,
        [
            [input_case1, 10, (4, 3), 2, (7, 8)],
            [input_case2, 4, (2, 3), 3, (4, 5)],
        ],
        'im2col_ext',
        disable_input_check=True
    )
