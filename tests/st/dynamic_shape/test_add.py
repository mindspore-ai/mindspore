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
from mindspore import context
from mindspore import ops
from mindspore import Tensor
import mindspore as ms

from tests.st.utils.test_utils import get_inputs_np, get_inputs_tensor, compare, to_cell_obj, run_with_cell
from tests.mark_utils import arg_mark

def get_np_data():
    return get_inputs_np([(2, 4, 8, 16), (2, 4, 8, 16)], [np.float16, np.float16])


@run_with_cell
def add_infervalue_func1():
    x = ms.Tensor(np.array([1, 2, 4]).astype(np.float32))
    y = ms.Tensor(np.array([2, 4, 3]).astype(np.float32))
    return ops.add(x, y)


@run_with_cell
def add_infervalue_func2():
    x = ms.Tensor(np.array([1, 2, 4]).astype(np.float32))
    y = ms.Tensor(np.array([3, 5, 1]).astype(np.float32))
    return ops.add(x, y)

@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_windows'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_add(mode):
    """
    Feature: Test add op.
    Description: Test add.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    inputs_np = get_np_data()
    func = ops.add
    input_args = get_inputs_tensor(inputs_np)
    # forward:
    output = func(*input_args)
    expect_output = np.add(*inputs_np)
    assert np.allclose(output.asnumpy(), expect_output)
    # backward:
    output_grads = ops.grad(func, grad_position=(0, 1))(*input_args)
    expect_grads = (np.ones_like(inputs_np[0]), np.ones_like(inputs_np[1]))
    compare(output_grads, expect_grads)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_windows'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_add_vmap(mode):
    """
    Feature: Test add op.
    Description: Test add vmap.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    inputs_np = get_np_data()
    x_batched, y_batched = inputs_np
    func = ops.add
    output_vmap = ops.vmap(func, in_axes=0)(*get_inputs_tensor([x_batched, y_batched]))
    output_batched = []
    for x, y in zip(x_batched, y_batched):
        output_batched.append(func(*get_inputs_tensor([x, y])).asnumpy())
    expect = np.stack(output_batched)
    assert np.allclose(output_vmap.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_windows'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_add_dyn(mode):
    """
    Feature: Test add op.
    Description: Test add dynamic shape.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    net = to_cell_obj(ops.add)
    net.set_inputs(Tensor(shape=[2, None, None, None], dtype=ms.float16),
                   Tensor(shape=[None, 4, None, None], dtype=ms.float16))
    case1_input_np = get_inputs_np([(2, 4, 8, 16), (2, 4, 8, 16)], [np.float16, np.float16])
    dout = np.ones_like(case1_input_np[0])
    case1_expect_grads = (dout, dout)
    case2_input_np = get_inputs_np([(2, 4, 1, 16), (1, 4, 8, 16)], [np.float16, np.float16])
    case2_expect_grads = (np.sum(dout, axis=2, keepdims=True), np.sum(dout, axis=0, keepdims=True))
    cases = [(case1_input_np, case1_expect_grads),
             (case2_input_np, case2_expect_grads)]
    for case in cases:
        input_case_np, expect_grads = case
        input_case_t = get_inputs_tensor(input_case_np)
        output = net(*input_case_t)
        expect = np.add(*input_case_np)
        compare(output, expect)
        output_grads = ops.grad(net, grad_position=(0, 1))(*input_case_t)
        compare(output_grads, expect_grads)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_add_op_infervalue(context_mode):
    """
    Feature: Ops.
    Description: test op add infervalue.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    out_1 = add_infervalue_func1()
    expect_out_1 = np.array([3., 6., 7.]).astype(np.float32)
    np.testing.assert_array_equal(out_1.asnumpy(), expect_out_1)
    out_2 = add_infervalue_func2()
    expect_out_2 = np.array([4., 7., 5.]).astype(np.float32)
    np.testing.assert_array_equal(out_2.asnumpy(), expect_out_2)
