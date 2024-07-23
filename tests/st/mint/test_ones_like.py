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
# pylint: disable=unused-variable
import pytest
import numpy as np
import mindspore as ms
from mindspore.common import dtype as mstype
from mindspore import ops, mint, Tensor, jit, JitConfig
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark


def ones_like_forward_func(input_tensor, dtype=None):
    y = mint.ones_like(input_tensor, dtype=dtype)
    return y


def ones_like_backward_func(input_tensor, dtype=None):
    input_grad = ops.grad(ones_like_forward_func, (0,))(input_tensor, dtype=dtype)
    return input_grad


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['GE', 'pynative', 'KBK'])
def test_ones_like_normal(mode):
    """
    Feature: Ops.
    Description: test ones_like.
    Expectation: expect correct result.
    """
    input_tensor = Tensor(np.arange(6).reshape(1, 2, 3), dtype=mstype.float32)
    dtype = None
    expect_y = np.ones((1, 2, 3), dtype=np.float32)

    dtype1 = mstype.int32
    expect_grad = 0
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        y = ones_like_forward_func(input_tensor, dtype)
        input_grad = ones_like_backward_func(input_tensor, dtype1)
    elif mode == 'KBK':
        y = (jit(ones_like_forward_func, jit_config=JitConfig(jit_level="O0")))(input_tensor, dtype)
        input_grad = (jit(ones_like_backward_func, jit_config=JitConfig(jit_level="O0")))(input_tensor, dtype1)
    else:
        y = (jit(ones_like_forward_func, jit_config=JitConfig(jit_level="O2")))(input_tensor, dtype)
        input_grad = (jit(ones_like_backward_func, jit_config=JitConfig(jit_level="O2")))(input_tensor, dtype1)
    np.testing.assert_allclose(y.asnumpy(), expect_y, rtol=1e-5)
    np.testing.assert_allclose(input_grad.asnumpy(), expect_grad, rtol=1e-5)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_ones_like_dynamic_shape():
    """
    Feature: Test ones_like with dynamic shape in graph mode.
    Description: call ops.extend.ones_like with valid input and index.
    Expectation: return the correct value.
    """
    tensor_1 = Tensor(np.arange(6).reshape(2, 3), dtype=mstype.float32)

    tensor_2 = Tensor(np.arange(24).reshape(2, 3, 4), dtype=mstype.float32)

    TEST_OP(ones_like_forward_func, [[tensor_1], [tensor_2]], '', disable_yaml_check=True)
