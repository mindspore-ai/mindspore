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
import numpy as np
import pytest
import mindspore as ms
from mindspore.nn import Cell
from mindspore.ops.extend import one_hot
from mindspore import ops
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark

def generate_random_input(num_classes):
    return np.random.permutation(num_classes)


@test_utils.run_with_cell
def onehot_forward_func(tensor, num_classes):
    return one_hot(tensor, num_classes)


def onehot_dyn_shape_func(tensor):
    return one_hot(tensor, 3)


def call_onehot(tensor, num_classes):
    """call_one_hot_ext"""
    out = ops.extend.one_hot(tensor, num_classes)
    return out

def generate_expect_backward_output():
    return 0

@test_utils.run_with_cell
def onehot_backward_func(tensor, num_classes):
    return ops.grad(onehot_forward_func, (0))(tensor, num_classes)

class Net(Cell):
    def __init__(self):
        super().__init__()
        self.one_hot = one_hot
    def construct(self, x, num_classes):
        return self.one_hot(x, num_classes)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_ops_onehot_forward1(mode):
    """
    Feature: pyboost function.
    Description: test function onehot forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.array([0, 1, 2]), ms.int64)
    output = onehot_forward_func(ms.Tensor(x), -1)
    expect_output = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]]).astype(np.int64)
    np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_onehot_forward(mode):
    """
    Feature: pyboost function.
    Description: test function onehot forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.array([0, 1, 2]), ms.int64)
    output = onehot_forward_func(ms.Tensor(x), 3)
    expect_output = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]]).astype(np.int64)
    np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_onehot_backward(mode):
    """
    Feature: pyboost function.
    Description: test function onehot backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = generate_random_input(2)
    output = onehot_backward_func(ms.Tensor(x), 3)
    expect = generate_expect_backward_output()
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_onehot_vmap(mode):
    """
    Feature: pyboost function.
    Description: test function onehot vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.array([0, 1, 2]), ms.int64)
    onehot_ext_vmap_func = ops.vmap(call_onehot, in_axes=(-1, None), out_axes=0)
    output = onehot_ext_vmap_func(ms.Tensor(x), 3)
    expect_output = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]]).astype(np.int64)
    np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-3)



@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_onehot_forward_dynamic_shape(mode):
    """
    Feature: pyboost function.
    Description: test function onehot forward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)

    x_dyn = ms.Tensor(shape=[None], dtype=ms.int64)
    test_cell = test_utils.to_cell_obj(onehot_dyn_shape_func)
    test_cell.set_inputs(x_dyn)

    expect_output = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]]).astype(np.int64)

    x1 = ms.Tensor(np.array([0, 1, 2]), ms.int64)
    output = test_cell(ms.Tensor(x1))
    np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-3)

    x2 = ms.Tensor(np.array([0, 1, 2]), ms.int64)
    output = test_cell(ms.Tensor(x2))
    np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-3)



@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_onehot_forward_dynamic_rank(mode):
    """
    Feature: pyboost function.
    Description: test function onehot forward with dynamic rank.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)

    x_dyn = ms.Tensor(shape=None, dtype=ms.int64)
    test_cell = test_utils.to_cell_obj(onehot_forward_func)
    test_cell.set_inputs(x_dyn, 3)

    expect_output = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]]).astype(np.int64)

    x1 = ms.Tensor(np.array([0, 1, 2]), ms.int64)
    output = test_cell(ms.Tensor(x1), 3)
    np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-3)

    x2 = ms.Tensor(np.array([0, 1, 2]), ms.int64)
    output = test_cell(ms.Tensor(x2), 3)
    np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_onehot_backward_dynamic_shape(mode):
    """
    Feature: pyboost function.
    Description: test function onehot backward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)

    x_dyn = ms.Tensor(shape=[None], dtype=ms.int64)
    test_cell = test_utils.to_cell_obj(onehot_backward_func)
    test_cell.set_inputs(x_dyn, 3)

    x1 = generate_random_input(2)
    output = test_cell(ms.Tensor(x1), 3)
    expect = generate_expect_backward_output()
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x2 = generate_random_input(2)
    output = test_cell(ms.Tensor(x2), 3)
    expect = generate_expect_backward_output()
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_onehot_backward_dynamic_rank(mode):
    """
    Feature: pyboost function.
    Description: test function onehot backward with dynamic rank.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)

    x_dyn = ms.Tensor(shape=None, dtype=ms.int64)
    test_cell = test_utils.to_cell_obj(onehot_backward_func)
    test_cell.set_inputs(x_dyn, 3)

    x1 = generate_random_input(2)
    output = test_cell(ms.Tensor(x1), 3)
    expect = generate_expect_backward_output()
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x2 = generate_random_input(2)
    output = test_cell(ms.Tensor(x2), 3)
    expect = generate_expect_backward_output()
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_onehot_dynamic_shape_testop():
    """
    Feature: Test onehot with dynamic shape in graph mode using TEST_OP.
    Description: call ops.onehot with valid input and index.
    Expectation: return the correct value.
    """
    x1 = generate_random_input(2)
    x2 = generate_random_input(2)

    TEST_OP(onehot_forward_func, [[ms.Tensor(x1), 3], [ms.Tensor(x2), 3]], '', disable_input_check=True,
            disable_yaml_check=True, disable_grad=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('param_jit_level', ["O2", "O0"])
def test_onehot_vmap(param_jit_level):
    """
    Feature: Test onehot with vmap.
    Description: call ops.extend.one_hot with valid input and index.
    Expectation: return the correct value.
    """
    def _foreach_run(inputs, num_classes, batch):
        out = []
        for i in range(inputs.shape[batch]):
            if batch == -1:
                input_inner = inputs[..., i]
            else:
                input_inner = inputs[i, ...]
            out.append(onehot_forward_func(input_inner, num_classes))
        out = ops.Stack()(out)
        return out

    ms.set_context(jit_level=param_jit_level)
    x = generate_random_input(2)

    batch_axis = -1
    onehot_ext_vmap_func = ops.vmap(call_onehot, in_axes=(batch_axis, None), out_axes=0)
    output = onehot_ext_vmap_func(ms.Tensor(x), 3)
    expect = _foreach_run(ms.Tensor(x), 3, batch_axis)
    assert np.allclose(output.asnumpy(), expect.asnumpy(), rtol=1e-4)

    batch_axis = 0
    onehot_ext_vmap_func = ops.vmap(call_onehot, in_axes=(batch_axis, None), out_axes=0)
    output = onehot_ext_vmap_func(ms.Tensor(x), 3)
    expect = _foreach_run(ms.Tensor(x), 3, batch_axis)
    assert np.allclose(output.asnumpy(), expect.asnumpy(), rtol=1e-4)



@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_one_hot_int32(mode):
    """
    Feature: ops.extend.one_hot
    Description: Verify the result of one_hot
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = ms.Tensor(np.array([0, 1, 2]), ms.int32)
    net = Net()
    output = net(x, 3)
    expect_output = [[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]]
    assert np.allclose(output.asnumpy(), expect_output)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_one_hot_int64(mode):
    """
    Feature: ops.extend.one_hot
    Description: Verify the result of one_hot
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = ms.Tensor(np.array([0, 1, 2]), ms.int64)
    net = Net()
    output = net(x, 3)
    expect_output = [[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]]
    assert np.allclose(output.asnumpy(), expect_output)
