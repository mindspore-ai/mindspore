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
from mindspore import ops
from mindspore.mint import ravel
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def ravel_func(x):
    return ravel(x)


@test_utils.run_with_cell
def ravel_forward_func(x):
    return ravel_func(x)


def ravel_bwd_func(x):
    return ops.grad(ravel_func, (0,))(x)


@test_utils.run_with_cell
def ravel_backward_func(x):
    return ravel_bwd_func(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ravel_normal(mode):
    """
    Feature: Ops.
    Description: test op ravel.
    Expectation: expect correct result.
    """
    ms.set_context(jit_level='O0')
    ms.set_context(mode=mode)
    test_shape = (2, 3, 4, 5)
    x = generate_random_input(test_shape, np.float32)
    output = ravel_forward_func(ms.Tensor(x))
    expect = x.flatten()
    assert output.is_contiguous()
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4)

    output = ravel_backward_func(ms.Tensor(x))
    expect = np.ones(test_shape).astype(np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.01)
@pytest.mark.parametrize("mode", [ms.PYNATIVE_MODE])
def test_ravel_uncontiguous(mode):
    """
    Feature: Ops.
    Description: test op ravel.
    Expectation: expect correct result.
    """
    ms.set_context(jit_level='O0')
    ms.set_context(mode=mode)
    test_shape = (2, 3, 4, 5)
    x = generate_random_input(test_shape, np.float32)
    ms_data = ms.Tensor(x)
    ms_data = ms_data.transpose()
    assert not ms_data.is_contiguous()
    output = ravel_forward_func(ms_data)
    expect = ms_data.flatten()
    assert output.is_contiguous()
    np.testing.assert_allclose(output.asnumpy(), expect.asnumpy(), rtol=1e-4)

    output = ravel_backward_func(ms_data)
    expect = np.ones(ms_data.shape).astype(np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=2048.01)
def test_ravel_dynamic_shape():
    """
    Feature: Test dynamic shape.
    Description: test function div dynamic feature.
    Expectation: expect correct result.
    """
    ms_data1 = generate_random_input((2, 3, 4), np.float32)
    ms_data2 = generate_random_input((3, 4, 5, 6), np.float32)
    TEST_OP(ravel_forward_func, [[ms.Tensor(ms_data1)], [ms.Tensor(ms_data2)]],
            disable_mode=['GRAPH_MODE_GE'])
