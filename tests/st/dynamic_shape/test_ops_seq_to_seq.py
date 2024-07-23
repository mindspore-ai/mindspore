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
from tests.st.utils import test_utils

from mindspore import ops, mutable, Tensor
import mindspore as ms
from mindspore.ops_generate.gen_ops_inner_prim import ListToTuple, TupleToList
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def list_to_tuple_forward_func(a, b, c, d, e):
    return ListToTuple()([a, b, c, d, e])


@test_utils.run_with_cell
def list_to_tuple_forward_dyn_func(x):
    return ListToTuple()(x)


@test_utils.run_with_cell
def list_to_tuple_backward_func(a, b, c, d, e):
    return ops.grad(list_to_tuple_forward_func, (0,))(a, b, c, d, e)


@test_utils.run_with_cell
def list_to_tuple_backward_dyn_func(x):
    return ops.grad(list_to_tuple_forward_dyn_func, (0,))(x)


@test_utils.run_with_cell
def tuple_to_list_forward_func(a, b, c, d, e):
    return TupleToList()((a, b, c, d, e))


@test_utils.run_with_cell
def tuple_to_list_forward_dyn_func(x):
    return TupleToList()(x)


@test_utils.run_with_cell
def tuple_to_list_backward_func(a, b, c, d, e):
    return ops.grad(tuple_to_list_forward_func, (0,))(a, b, c, d, e)


@test_utils.run_with_cell
def tuple_to_list_backward_dyn_func(x):
    return ops.grad(tuple_to_list_forward_dyn_func, (0,))(x)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.context.PYNATIVE_MODE, ms.context.GRAPH_MODE])
def test_seq_to_seq_forward(mode):
    """
    Feature: Ops.
    Description: test op ListToTuple and TupleToList by constant input.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode, device_target='CPU')
    input_scalar = 1.5
    input_list = [1, 2, 3]
    input_tuple = (1, 2, 3)
    input_dict = {"key1": 1}
    input_float32_tensor = ms.Tensor([1.1], ms.float32)
    output = list_to_tuple_forward_func(input_scalar, input_list, input_tuple,\
                                        input_dict, input_float32_tensor)
    assert isinstance(output, tuple)
    expect = [input_scalar, input_list, input_tuple, input_dict, input_float32_tensor]
    for out, exp in zip(output, expect):
        if isinstance(exp, dict) and mode == ms.GRAPH_MODE:
            assert out == (1,)
        else:
            assert out == exp

    output = tuple_to_list_forward_func(input_scalar, input_list, input_tuple,\
                                        input_dict, input_float32_tensor)
    assert isinstance(output, list)
    expect = [input_scalar, input_list, input_tuple, input_dict, input_float32_tensor]
    for out, exp in zip(output, expect):
        if isinstance(exp, dict) and mode == ms.GRAPH_MODE:
            assert out == (1,)
        else:
            assert out == exp


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.context.PYNATIVE_MODE, ms.context.GRAPH_MODE])
def test_seq_to_seq_forward_dyn(mode):
    """
    Feature: Ops.
    Description: test op ListToTuple and TupleToList by mutable input.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode, device_target='CPU')
    expect1 = (1, 2, 3)
    input_x1 = mutable([1, 2, 3], True)
    output1 = list_to_tuple_forward_dyn_func(input_x1)
    assert isinstance(output1, tuple)
    assert np.allclose(output1, expect1)
    expect2 = [Tensor([1]), Tensor([2]), Tensor([3])]
    input_x2 = mutable([Tensor([1]), Tensor([2]), Tensor([3])], True)
    output2 = list_to_tuple_forward_dyn_func(input_x2)
    assert isinstance(output2, tuple)
    for out2, exp2 in zip(output2, expect2):
        assert np.allclose(out2.asnumpy(), exp2.asnumpy())

    expect1 = [1, 2, 3]
    input_x1 = mutable((1, 2, 3), True)
    output1 = tuple_to_list_forward_dyn_func(input_x1)
    assert isinstance(output1, list)
    assert np.allclose(output1, expect1)
    expect2 = [Tensor([1]), Tensor([2]), Tensor([3])]
    input_x2 = mutable((Tensor([1]), Tensor([2]), Tensor([3])), True)
    output2 = tuple_to_list_forward_dyn_func(input_x2)
    assert isinstance(output2, list)
    for out2, exp2 in zip(output2, expect2):
        assert np.allclose(out2.asnumpy(), exp2.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.context.PYNATIVE_MODE, ms.context.GRAPH_MODE])
def test_seq_to_seq_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op ListToTuple and TupleToList by constant input.
    Expectation: empty tuple.
    """
    ms.context.set_context(mode=mode, device_target='CPU')

    class Net(ms.nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.tuple_to_list = TupleToList()
            self.list_to_tuple = ListToTuple()

        def construct(self, x):
            return self.list_to_tuple(self.tuple_to_list(x))

    class GradNetWrtX(ms.nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net

        def construct(self, x):
            gradient_function = ms.grad(self.net)
            return gradient_function(x)

    input_scalar = 1.5
    input_list = [1, 2, 3]
    input_tuple = (1, 2, 3)
    input_dict = {0: Tensor([0]), 1: Tensor([1])}
    input_float32_tensor = ms.Tensor([1.1], ms.float32)
    grads = GradNetWrtX(Net())((input_scalar, input_list, input_tuple,\
                                         input_dict, input_float32_tensor))
    assert grads == ()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE])
def test_seq_to_seq_backward_dyn(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op ListToTuple and TupleToList by mutable input.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode, device_target='CPU')
    expect = (1, 1, 1)
    input_x1 = mutable([1, 2, 3], True)
    grads1 = list_to_tuple_backward_dyn_func(input_x1)
    assert np.allclose(grads1, expect)
    input_x2 = mutable([Tensor([1]), Tensor([2]), Tensor([3])], True)
    grads2 = list_to_tuple_backward_dyn_func(input_x2)
    for out2, exp2 in zip(grads2, expect):
        assert np.allclose(out2.asnumpy(), exp2)

    expect = (1, 1, 1)
    input_x = mutable((1, 2, 3), True)
    grads = tuple_to_list_backward_dyn_func(input_x)
    assert np.allclose(grads, expect)
    input_x2 = mutable((Tensor([1]), Tensor([2]), Tensor([3])), True)
    grads2 = tuple_to_list_backward_dyn_func(input_x2)
    for out2, exp2 in zip(grads2, expect):
        assert np.allclose(out2.asnumpy(), exp2)
