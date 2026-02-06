# Copyright 2023 Huawei Technoflipies Co., Ltd
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
from mindspore import Tensor, context
from mindspore.ops import flip
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(input_x, dims):
    return np.flip(input_x, dims)


def generate_expect_backward_output(input_x, dims):
    return np.ones_like(input_x)


@test_utils.run_with_cell
def flip_forward_func(input_x, dims):
    return flip(input_x, dims)


@test_utils.run_with_cell
def flip_backward_func(input_x, dims):
    return ms.grad(flip_forward_func, (0))(input_x, dims)


@test_utils.run_with_cell
def flip_vmap_func(input_x, dims):
    return ms.ops.vmap(flip_forward_func, in_axes=(0, None), out_axes=(0))(input_x, dims)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.02)
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_flip_normal(mode):
    """
    Feature: test flip operator
    Description: test flip run by pyboost
    Expectation: success
    """
    context.set_context(mode=mode)
    np_array = np.random.rand(2, 3, 4)
    input_x = Tensor(np_array, ms.float32)
    dims = (0, 1)
    output = flip_forward_func(input_x, dims)
    expect = generate_expect_forward_output(np_array, dims)
    assert np.allclose(output.asnumpy(), expect)

    output = flip_backward_func(input_x, dims)
    expect = generate_expect_backward_output(np_array, dims)
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_flip_vmap(mode):
    """
    Feature: pyboost function.
    Description: test function flip vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    input_x = generate_random_input((1, 2, 3), np.float32)
    dims = [1]
    output = flip_vmap_func(ms.Tensor(input_x), dims)
    dims1 = [2]
    expect = generate_expect_forward_output(input_x, dims1)
    np.testing.assert_allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=3072.01)
def test_flip_dynamic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test ops.flip dynamic shape feature.
    Expectation: expect correct result.o
    """
    input_case1 = Tensor(np.random.rand(3, 4, 5, 6).astype(np.float32))
    input_case2 = Tensor(np.random.rand(3, 4).astype(np.float32))
    TEST_OP(flip_forward_func, [[input_case1, (0, -1)], [input_case2, (-1, 0)]],
            disable_case=['ScalarTensor'],
            case_config={'disable_input_check': True})
