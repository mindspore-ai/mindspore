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
from mindspore import ops, Tensor
from mindspore.mint import squeeze
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

@test_utils.run_with_cell
def squeeze_forward_func(x, dim):
    return squeeze(x, dim)


@test_utils.run_with_cell
def squeeze_backward_func(x, dim):
    return ms.grad(squeeze_forward_func, (0))(x, dim)


@test_utils.run_with_cell
def squeeze_vmap_func(x, dim):
    return ops.vmap(squeeze_forward_func, in_axes=(0, None), out_axes=0)(x, dim)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_squeeze_normal(context_mode):
    """
    Feature: pyboost function.
    Description: test function squeeze forward and backward.
    Expectation: expect correct result.
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)
    # forward
    # int
    x = Tensor(np.array([[[2, 2], [2, 2]]]), ms.float32)
    dim = 0
    out = squeeze_forward_func(x, dim)
    expect = [[2., 2.], [2., 2.]]
    np.testing.assert_allclose(out.asnumpy(), expect, rtol=1e-3)
    # tuple(ints)
    x1 = Tensor(np.array([[[[2, 2]], [[2, 2]]]]), ms.float32)
    dim1 = (0, -2)
    out1 = squeeze_forward_func(x1, dim1)
    expect1 = [[2., 2.], [2., 2.]]
    np.testing.assert_allclose(out1.asnumpy(), expect1, rtol=1e-3)

    # backward
    output_b = squeeze_backward_func(ms.Tensor(x), dim)
    expect_b = [[[1., 1.], [1., 1.]]]
    np.testing.assert_allclose(output_b.asnumpy(), expect_b, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_squeeze_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function squeeze vmap feature.
    Expectation: expect correct result.
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)
    x = Tensor(np.array([[[[[2., 2.], [2., 2.]]]]]), ms.float32)
    dim = 0
    output = squeeze_vmap_func(x, dim)
    expect = [[[[2., 2.], [2., 2.]]]]
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
def test_ops_squeeze_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function squeeze with dynamic shape and dynamic rank.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((1, 2, 3, 4, 5), np.float32)
    dim1 = 0
    x2 = generate_random_input((4, 1, 5), np.float32)
    dim2 = 1
    TEST_OP(squeeze_forward_func, [[ms.Tensor(x1), dim1], [ms.Tensor(x2), dim2]],
            disable_mode=["GRAPH_MODE_GE"],
            disable_case=['ScalarTensor'])
