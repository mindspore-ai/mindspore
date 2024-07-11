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
import mindspore as ms
from mindspore.common import dtype as mstype
from mindspore import ops, mint, Tensor, jit, JitConfig
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def median_forward_func(input_tensor, axis=None, keepdim=None):
    return mint.median(input_tensor, axis, keepdim)


@test_utils.run_with_cell
def median_backward_func(input_tensor, axis=None, keepdim=None):
    input_grad = ops.grad(median_forward_func, (0,))(input_tensor, axis, keepdim)
    return input_grad


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_median_dim_normal(mode):
    """
    Feature: test aclnnMedianDim and aclnnMedian.
    Description: test median_ext.
    Expectation: expect correct result or success.
    """

    input_tensor_list = [
        Tensor([[2, 4, 1, 5, 6], [0, 4, 1, 9, 2.0]], dtype=mstype.float32),
        Tensor([[2, 4, 1, 5, 6], [0, 4, 1, 9, 2.0]], dtype=mstype.float32)
    ]
    grad_np_list = [
        np.array([[0., 1., 0., 0., 0.], [0., 0., 0., 0., 1.]]),
        np.array([[0.5, 0., 0., 0., 0.], [0., 0., 0., 0., 0.5]])
    ]
    axis_list = [1, None]
    keepdim_list = [False, None]
    expect_y_list = [
        np.array([4, 2]),
        np.array(2)
    ]
    expect_indices_list = [
        np.array([1, 4]),
        None
    ]

    for i in range(len(input_tensor_list)):
        input_tensor = input_tensor_list[i]
        grad_np = grad_np_list[i]
        axis = axis_list[i]
        keepdim = keepdim_list[i]
        expect_y = expect_y_list[i]
        expect_indices = expect_indices_list[i]
        if mode == 'pynative':
            ms.context.set_context(mode=ms.PYNATIVE_MODE)
            y = median_forward_func(input_tensor, axis, keepdim)
            grad = median_backward_func(input_tensor, axis, keepdim)
        elif mode == 'KBK':
            y = (jit(median_forward_func, jit_config=JitConfig(jit_level="O0")))(input_tensor, axis, keepdim)
            grad = (jit(median_backward_func, jit_config=JitConfig(jit_level="O0")))(input_tensor, axis, keepdim)
        else:
            y = (jit(median_forward_func, jit_config=JitConfig(jit_level="O2")))(input_tensor, axis, keepdim)
            grad = (jit(median_backward_func, jit_config=JitConfig(jit_level="O2")))(input_tensor, axis, keepdim)
        if isinstance(y, tuple):
            np.testing.assert_allclose(y[0].asnumpy(), expect_y, rtol=1e-5)
            np.testing.assert_allclose(y[1].asnumpy(), expect_indices, rtol=1e-5)
            np.testing.assert_allclose(grad.asnumpy(), grad_np, rtol=1e-5)
        else:
            np.testing.assert_allclose(y.asnumpy(), expect_y, rtol=1e-5)
            np.testing.assert_allclose(grad.asnumpy(), grad_np, rtol=1e-5)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_median_dynamic_shape():
    """
    Feature: Test meidan with dynamic shape in graph mode.
    Description: call mint.median with valid input and index.
    Expectation: return the correct value.
    """
    tensor_1 = Tensor(np.arange(6).reshape(2, 3), dtype=mstype.float32)
    tensor_2 = Tensor(np.arange(24).reshape(2, 3, 4), dtype=mstype.float32)

    TEST_OP(median_forward_func, [[tensor_1, 0, False], [tensor_2, 1, True]], '', disable_yaml_check=True,
            disable_mode=['GRAPH_MODE'])
