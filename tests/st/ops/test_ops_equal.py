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
import mindspore as ms
from mindspore import ops
from mindspore.ops.auto_generate import equal

import tests.st.utils.test_utils as test_utils
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype), np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x, other):
    return np.equal(x, other)


@test_utils.run_with_cell
def equal_forward_func(x, other):
    return equal(x, other)


@test_utils.run_with_cell
def equal_backward_func(x, other):
    return ops.grad(equal_forward_func, (0, 1))(x, other)


@test_utils.run_with_cell
def equal_vmap_func(x, other):
    return ops.vmap(equal_forward_func)(x, other)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_equal_normal(context_mode):
    """
    Feature: pyboost function.
    Description: test function equal forward and backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x, other = generate_random_input((2, 3, 4, 5), np.float32)
    output = equal_forward_func(ms.Tensor(x), ms.Tensor(other))
    expect = generate_expect_forward_output(x, other)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    equal_backward_func(ms.Tensor(x), ms.Tensor(other))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_equal_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function equal vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x, other = generate_random_input((2, 3, 4, 5), np.float32)
    output = equal_vmap_func(ms.Tensor(x), ms.Tensor(other))
    expect = generate_expect_forward_output(x, other)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_equal_forward_dynamic_shape(context_mode):
    """
    Feature: pyboost function.
    Description: test function equal forward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    y_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(equal_forward_func)
    test_cell.set_inputs(x_dyn, y_dyn)
    x1, other1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1), ms.Tensor(other1))
    expect = generate_expect_forward_output(x1, other1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
    x2, other2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2), ms.Tensor(other2))
    expect = generate_expect_forward_output(x2, other2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_equal_forward_dynamic_rank(context_mode):
    """
    Feature: pyboost function.
    Description: test function equal forward with dynamic rank.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    y_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(equal_forward_func)
    test_cell.set_inputs(x_dyn, y_dyn)
    x1, other1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1), ms.Tensor(other1))
    expect = generate_expect_forward_output(x1, other1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
    x2, other2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2), ms.Tensor(other2))
    expect = generate_expect_forward_output(x2, other2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


class Net1(ms.nn.Cell):
    def construct(self, a, b, start=None, end=None, step=None):
        a[start:end:step] = b
        return a


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE,])
def test_a_is_variable_list_b_is_list_or_tuple(context_mode):
    """
    Feature: DT test: Graph mode Parameter[0] has not default param
    Description: test equal result.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    net = Net1()
    x = ms.Tensor(1)
    ma = net(a=[x, x, x, x, x], b=[11, 22, 33], start=None, end=1, step=None)
    pa = [11, 22, 33, ms.Tensor(1), ms.Tensor(1), ms.Tensor(1), ms.Tensor(1)]
    assert ma == pa

    x = ms.Tensor(1)
    ma = net(a=[x, x, x, x, x], b=[11, 22, 33], start=2, end=3, step=None)
    pa = [ms.Tensor(1), ms.Tensor(1), ms.Tensor(11), ms.Tensor(22), ms.Tensor(33), ms.Tensor(1), ms.Tensor(1)]
    assert ma == pa

    ma = net(a=[x, x, x, x, x], b=(11, 22, 33), start=-1, end=None, step=None)
    pa = [ms.Tensor(1), ms.Tensor(1), ms.Tensor(1), ms.Tensor(1), 11, 22, 33]
    assert ma == pa

    ma = net(a=[x, x, x, x, x], b=[ms.Tensor(11), ms.Tensor(22), ms.Tensor(33)], start=None, end=None, step=2)
    pa = [ms.Tensor(11), ms.Tensor(1), ms.Tensor(22), ms.Tensor(1), ms.Tensor(33)]
    assert ma == pa

    ma = net(a=[x, x, x, x, x], b=[11, 22, 33], start=None, end=None, step=-2)
    pa = [33, ms.Tensor(1), 22, ms.Tensor(1), 11]
    assert ma == pa

    with pytest.raises(ValueError):
        net(a=[x, x, x, x, x], b=(11, 22, 33), start=-1, end=None, step=3)
