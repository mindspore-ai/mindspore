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
import mindspore as ms
from mindspore import ops, jit, JitConfig
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x):
    return np.any(x)




@test_utils.run_with_cell
def any_forward_func(x):
    return ms.ops.any(x)



@test_utils.run_with_cell
def any_vmap_func(x, in_axes=0):
    return ops.vmap(any_forward_func, in_axes, out_axes=0)(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_any_forward(context_mode):
    """
    Feature: pyboost function.
    Description: test function any forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = any_forward_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)



@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_any_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function any vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = any_vmap_func(ms.Tensor(x), 0)
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_any_forward_dynamic_shape(context_mode):
    """
    Feature: pyboost function.
    Description: test function any forward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(any_forward_func)
    test_cell.set_inputs(x_dyn)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1))
    expect = generate_expect_forward_output(x1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2))
    expect = generate_expect_forward_output(x2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_any_forward_dynamic_rank(context_mode):
    """
    Feature: pyboost function.
    Description: test function any forward with dynamic rank.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(any_forward_func)
    test_cell.set_inputs(x_dyn)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1))
    expect = generate_expect_forward_output(x1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2))
    expect = generate_expect_forward_output(x2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)




@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK', 'GE'])
def test_any_forward_static_shape(mode):
    """
    Feature: Test any with static shape in graph and pynative mode.
    Description: call ops.any with valid input and index.
    Expectation: return the correct value.
    """
    x = generate_random_input((3, 4, 5), np.float32)

    if mode == 'pynative':
        output = any_forward_func(ms.Tensor(x))
    elif mode == 'KBK':
        output = (jit(any_forward_func, jit_config=JitConfig(jit_level="O0")))(ms.Tensor(x))
    else:
        output = (jit(any_forward_func, jit_config=JitConfig(jit_level="O2")))(ms.Tensor(x))

    expect = generate_expect_forward_output(x)
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4)



@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_any_dynamic_shape_testop():
    """
    Feature: Test any with dynamic shape in graph mode using TEST_OP.
    Description: call ops.any with valid input and index.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((3, 4, 5), np.float32)
    x2 = generate_random_input((3, 7, 8, 3), np.float32)

    TEST_OP(any_forward_func, [[ms.Tensor(x1)], [ms.Tensor(x2)]], '', disable_yaml_check=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('param_jit_level', ["O2", "O0"])
def test_any_vmap(param_jit_level):
    """
    Feature: Test any with vmap.
    Description: call ops.any with valid input and index.
    Expectation: return the correct value.
    """
    def _foreach_run(inputs, batch):
        out = []
        for i in range(inputs.shape[batch]):
            if batch == -1:
                input_inner = inputs[..., i]
            else:
                input_inner = inputs[i, ...]
            out.append(any_forward_func(input_inner))
        out = ops.Stack()(out)
        return out

    ms.set_context(jit_level=param_jit_level)
    x = generate_random_input((4, 5, 6), np.float32)

    batch_axis = -1
    output = any_vmap_func(ms.Tensor(x), batch_axis)
    expect = _foreach_run(ms.Tensor(x), batch_axis)
    assert np.allclose(output.asnumpy(), expect.asnumpy(), rtol=1e-4)

    batch_axis = 0
    output = any_vmap_func(ms.Tensor(x), batch_axis)
    expect = _foreach_run(ms.Tensor(x), batch_axis)
    assert np.allclose(output.asnumpy(), expect.asnumpy(), rtol=1e-4)
    