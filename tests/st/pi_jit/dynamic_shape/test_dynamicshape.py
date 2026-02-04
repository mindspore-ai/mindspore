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
"""run dynamic shape test"""
import pytest
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, jit, context, Symbol, Parameter, ops
from mindspore.nn import Cell
from mindspore._c_expression import get_code_extra
from tests.st.pi_jit.share.utils import match_array, assert_equal, pi_jit_with_config
from tests.st.pi_jit.share.grad import GradOfAllInputs
from tests.mark_utils import arg_mark

s = Symbol(max=10, min=1)
g_relu = nn.ReLU()


@pytest.mark.skip(reason="Need to implement dynamic arg for jit api.")
@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
def test_dynamic_shape_case():
    """
    Feature: Method DynamicShape Testing
    Description: Test dyanmicshape function to check whether it works.
    Expectation: The result of the case should dump the dynamic shape ir at last.
                 'enable_dynamic_shape' flag is used to enable dynamic shape when calling 3 times for different shape.
    """
    @jit(capture_mode="bytecode", jit_config={"enable_dynamic_shape": True, "limit_graph_count": 1})
    def dynamic_shape_test(a, b):
        return a + b

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1])
    b = Tensor([2])
    expect = Tensor([3])
    c = dynamic_shape_test(a, b)
    assert all(c == expect)
    a = Tensor([1, 1])
    b = Tensor([2, 2])
    expect = Tensor([3, 3])
    c = dynamic_shape_test(a, b)
    assert all(c == expect)
    a = Tensor([1, 1, 1])
    b = Tensor([2, 2, 2])
    expect = Tensor([3, 3, 3])
    c = dynamic_shape_test(a, b)
    assert all(c == expect)
    a = Tensor([1, 1, 1, 1])
    b = Tensor([2, 2, 2, 2])
    expect = Tensor([3, 3, 3, 3])
    c = dynamic_shape_test(a, b)
    assert all(c == expect)
    a = Tensor([1, 1, 1, 1, 1])
    b = Tensor([2, 2, 2, 2, 2])
    expect = Tensor([3, 3, 3, 3, 3])
    c = dynamic_shape_test(a, b)
    assert all(c == expect)
    a = Tensor([1, 1, 1, 1, 1, 1])
    b = Tensor([2, 2, 2, 2, 2, 2])
    expect = Tensor([3, 3, 3, 3, 3, 3])
    c = dynamic_shape_test(a, b)
    assert all(c == expect)
    jcr = get_code_extra(dynamic_shape_test.__wrapped__)
    # when cnt=2>limit_graph_count=1, trigger gc and compile_count_ is 1(dynamic_shape) + 2 = 3
    assert jcr["compile_count_"] == 3

@pytest.mark.skip(reason="adapter later")
@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
def test_signature_case():
    """
    Feature: Method DynamicShape DynamicSymbolic In Signature Testing
    Description: Test dynamicshape and dynamicsymbolic in signature function to check whether it works.
    Expectation: The result of the case should compile the graph no more than once.
    """

    class SignatureNet(Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        @jit(capture_mode="bytecode", input_signature=(Tensor(shape=(s,None), dtype=ms.float32)))
        def construct(self, a):
            return self.relu(a)

    @jit(capture_mode="bytecode", input_signature=(Tensor(shape=(None, s), dtype=ms.float32)))
    def signature_test(a):
        return g_relu(a)

    @jit(capture_mode="bytecode", input_signature=((Tensor(shape=(None, s), dtype=ms.float32), Tensor(shape=(None, s), dtype=ms.float32)), None))
    def signature_tuple_test(a, b):
        return g_relu(a[0])

    context.set_context(mode=context.PYNATIVE_MODE)
    t1 = Tensor([[1.1, 1.1],[2.2,2.2]], dtype=ms.float32)
    t2 = Tensor([[1.1],[2.2]], dtype=ms.float32)
    res1 = signature_test(t1)
    match_array(res1, t1)
    res2 = signature_test(t2)
    match_array(res2, t2)
    res1 = signature_tuple_test((t1, t2), 1)
    match_array(res1, t1)
    res2 = signature_tuple_test((t2, t1), 1)
    match_array(res2, t2)
    res1 = SignatureNet()(t1)
    match_array(res1, t1)
    res2 = SignatureNet()(t2)
    match_array(res2, t2)
    jcr1 = get_code_extra(signature_test.__wrapped__)
    assert(jcr1["compile_count_"] == 1)
    jcr2 = get_code_extra(SignatureNet().construct.__wrapped__)
    assert(jcr2["compile_count_"] == 1)


config = {"enable_dynamic_shape": True, "limit_graph_count": 1}


def _apply_jit(net: Cell, *, jit_config=None) -> Cell:
    if jit_config:
        config.update(jit_config)
    net.construct = pi_jit_with_config(net.construct, jit_config=config)
    return net


def _compute_grad_of_net_inputs(net: Cell, *inputs):
    grad_net = GradOfAllInputs(net, sens_param=False)
    grad_net.set_train()
    return grad_net(*inputs)


class ScalingNet(Cell):
    def __init__(self):
        super().__init__()
        self.factor = 2

    def construct(self, x):
        return self.factor * x


class MultiplySquareNet(Cell):
    def __init__(self):
        super().__init__()
        self.factor = 2

    def construct(self, x, y):
        return x * ops.square(y) * self.factor


class TupleSelectorNet(Cell):
    def __init__(self, index=0):
        super().__init__()
        self.index = index

    def construct(self, values):
        return values[self.index] + values[1]


class AssignAddNet(Cell):
    def __init__(self):
        super().__init__()
        self.factor = 2

    def construct(self, x, y):
        ops.assign_add(x, y)
        return ops.square(y) * self.factor


class EyeNet(Cell):
    def __init__(self):
        super().__init__()
        self.axis = 0

    def construct(self, x, m):
        size = x.shape[self.axis]
        y = ops.eye(size, m)
        return x * y


class OnesNet(Cell):
    def __init__(self):
        super().__init__()
        self.factor = 2

    def construct(self, shape):
        return ops.ones(shape, ms.float32) * self.factor


class TensorScalarNet(Cell):
    def __init__(self):
        super().__init__()
        self.factor = 2

    def construct(self, x, y):
        return x * y * self.factor


class MultiOutputNet(Cell):
    def __init__(self):
        super().__init__()
        self.factor = 2

    def construct(self, x, y):
        return self.factor * x, 3 * y


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_shape_scale_one_dimension():
    """
    Feature: Dynamic shape scaling.
    Description: Run the scaling cell with dynamic shape enabled while varying a single input dimension.
    Expectation: JIT forward outputs and gradients match pynative for every input.
    Migrated from: test_dynamic_shape_pijit.py::test_dynamic_shape_pijit_change_one_dim
    """
    shapes = [(2, 3), (2, 4), (2, 5)]
    for shape in shapes:
        x = Tensor(np.random.rand(*shape).astype(np.float32))

        pynative_net = ScalingNet()
        pynative_net.set_grad()
        pynative_output = pynative_net(x)
        pynative_grad = _compute_grad_of_net_inputs(pynative_net, x)

        jit_net = ScalingNet()
        jit_net.set_grad()
        jit_net = _apply_jit(jit_net)
        jit_output = jit_net(x)
        jit_grad = _compute_grad_of_net_inputs(jit_net, x)

        match_array(jit_output, pynative_output)
        assert_equal(jit_grad, pynative_grad, decimal=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_shape_scale_all_dimensions():
    """
    Feature: Dynamic shape scaling.
    Description: Run the scaling cell with dynamic shape enabled while changing all input dimensions each time.
    Expectation: JIT forward outputs and gradients match pynative for every shape.
    Migrated from: test_dynamic_shape_pijit.py::test_dynamic_shape_pijit_change_all_dim
    """
    shapes = [(2, 3), (3, 4), (4, 5)]
    for shape in shapes:
        x = Tensor(np.random.rand(*shape).astype(np.float32))

        pynative_net = ScalingNet()
        pynative_net.set_grad()
        pynative_output = pynative_net(x)
        pynative_grad = _compute_grad_of_net_inputs(pynative_net, x)

        jit_net = ScalingNet()
        jit_net.set_grad()
        jit_net = _apply_jit(jit_net)
        jit_output = jit_net(x)
        jit_grad = _compute_grad_of_net_inputs(jit_net, x)

        match_array(jit_output, pynative_output)
        assert_equal(jit_grad, pynative_grad, decimal=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_shape_scale_rank_variation():
    """
    Feature: Dynamic shape scaling.
    Description: Run the scaling cell while increasing the input rank across executions.
    Expectation: JIT forward outputs and gradients match pynative for every shape.
    Migrated from: test_dynamic_shape_pijit.py::test_dynamic_shape_pijit_change_rank
    """
    shapes = [(2, 3), (2, 3, 4), (2, 3, 4, 5)]
    for shape in shapes:
        x = Tensor(np.random.rand(*shape).astype(np.float32))

        pynative_net = ScalingNet()
        pynative_net.set_grad()
        pynative_output = pynative_net(x)
        sens = Tensor(np.random.randn(*pynative_output.shape).astype(np.float32))
        pynative_grad = _compute_grad_of_net_inputs(pynative_net, x)

        jit_net = ScalingNet()
        jit_net.set_grad()
        jit_net = _apply_jit(jit_net)
        jit_output = jit_net(x)
        jit_grad = _compute_grad_of_net_inputs(jit_net, x)

        match_array(jit_output, pynative_output)
        assert_equal(jit_grad, pynative_grad, decimal=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_shape_scale_mixed_variation():
    """
    Feature: Dynamic shape scaling.
    Description: Run the scaling cell while varying input dimensions and rank in the same session.
    Expectation: JIT forward outputs and gradients match pynative for every shape.
    Migrated from: test_dynamic_shape_pijit.py::test_dynamic_shape_pijit_change_shape_rank
    """
    shapes = [(2, 3), (2, 4), (2, 5), (2, 3, 4)]
    for shape in shapes:
        x = Tensor(np.random.rand(*shape).astype(np.float32))

        pynative_net = ScalingNet()
        pynative_net.set_grad()
        pynative_output = pynative_net(x)
        sens = Tensor(np.random.randn(*pynative_output.shape).astype(np.float32))
        pynative_grad = _compute_grad_of_net_inputs(pynative_net, x)

        jit_net = ScalingNet()
        jit_net.set_grad()
        jit_net = _apply_jit(jit_net)
        jit_output = jit_net(x)
        jit_grad = _compute_grad_of_net_inputs(jit_net, x)

        match_array(jit_output, pynative_output)
        assert_equal(jit_grad, pynative_grad, decimal=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_shape_scale_dtype_variation():
    """
    Feature: Dynamic shape scaling.
    Description: Run the scaling cell while varying input dtype across executions.
    Expectation: JIT forward outputs match pynative for all dtypes and gradients match for floating-point inputs.
    Migrated from: test_dynamic_shape_pijit.py::test_dynamic_shape_pijit_change_dtype
    """
    dtypes = [ms.float32, ms.float16, ms.int32]
    for dtype in dtypes:
        if dtype == ms.int32:
            data = np.random.randint(-5, 5, size=(2, 3)).astype(np.int32)
        elif dtype == ms.float16:
            data = np.random.rand(2, 3).astype(np.float16)
        else:
            data = np.random.rand(2, 3).astype(np.float32)
        x = Tensor(data, dtype)

        pynative_net = ScalingNet()
        pynative_net.set_grad()
        pynative_output = pynative_net(x)

        jit_net = ScalingNet()
        jit_net.set_grad()
        jit_net = _apply_jit(jit_net)
        jit_output = jit_net(x)

        match_array(jit_output, pynative_output)

        sens = Tensor(np.random.randn(*pynative_output.shape).astype(np.float32))
        pynative_grad = _compute_grad_of_net_inputs(pynative_net, x)
        jit_grad = _compute_grad_of_net_inputs(jit_net, x)
        assert_equal(jit_grad, pynative_grad, decimal=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_shape_two_inputs_change_first():
    """
    Feature: Dynamic shape with two inputs.
    Description: Change only the first input tensor shape across executions.
    Expectation: JIT forward outputs and gradients match pynative for every input pair.
    Migrated from: test_dynamic_shape_pijit.py::test_dynamic_shape_pijit_change_one_tensor
    """
    input_pairs = [
        (Tensor(np.random.rand(2, 3, 4).astype(np.float32)), Tensor(np.random.rand(2, 3, 4).astype(np.float32))),
        (Tensor(np.random.rand(2, 3, 1).astype(np.float32)), Tensor(np.random.rand(2, 3, 4).astype(np.float32))),
        (Tensor(np.random.rand(1, 3, 4).astype(np.float32)), Tensor(np.random.rand(2, 3, 4).astype(np.float32))),
    ]
    for x, y in input_pairs:
        pynative_net = MultiplySquareNet()
        pynative_net.set_grad()
        pynative_output = pynative_net(x, y)
        sens = Tensor(np.random.randn(*pynative_output.shape).astype(np.float32))
        pynative_grads = _compute_grad_of_net_inputs(pynative_net, x, y)

        jit_net = MultiplySquareNet()
        jit_net.set_grad()
        jit_net = _apply_jit(jit_net)
        jit_output = jit_net(x, y)
        jit_grads = _compute_grad_of_net_inputs(jit_net, x, y)

        match_array(jit_output, pynative_output)
        assert_equal(jit_grads, pynative_grads, decimal=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_shape_two_inputs_change_both():
    """
    Feature: Dynamic shape with two inputs.
    Description: Change both input tensor shapes across executions.
    Expectation: JIT forward outputs and gradients match pynative for every input pair.
    Migrated from: test_dynamic_shape_pijit.py::test_dynamic_shape_pijit_change_all_tensor
    """
    input_pairs = [
        (Tensor(np.random.rand(2, 3, 4).astype(np.float32)), Tensor(np.random.rand(2, 3, 4).astype(np.float32))),
        (Tensor(np.random.rand(2, 3, 1).astype(np.float32)), Tensor(np.random.rand(2, 3, 4).astype(np.float32))),
        (Tensor(np.random.rand(2, 3, 4).astype(np.float32)), Tensor(np.random.rand(2, 1, 4).astype(np.float32))),
    ]
    for x, y in input_pairs:
        pynative_net = MultiplySquareNet()
        pynative_net.set_grad()
        pynative_output = pynative_net(x, y)
        sens = Tensor(np.random.randn(*pynative_output.shape).astype(np.float32))
        pynative_grads = _compute_grad_of_net_inputs(pynative_net, x, y)

        jit_net = MultiplySquareNet()
        jit_net.set_grad()
        jit_net = _apply_jit(jit_net)
        jit_output = jit_net(x, y)
        jit_grads = _compute_grad_of_net_inputs(jit_net, x, y)

        match_array(jit_output, pynative_output)
        assert_equal(jit_grads, pynative_grads, decimal=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_shape_two_inputs_multiple_cases():
    """
    Feature: Dynamic shape with two inputs.
    Description: Run multiple shape combinations for both tensors to cover mixed dynamic cases.
    Expectation: JIT forward outputs and gradients match pynative for every input pair.
    Migrated from: test_dynamic_shape_pijit.py::test_dynamic_shape_pijit_change_two_tensor
    """
    input_pairs = [
        (Tensor(np.random.rand(2, 3, 4).astype(np.float32)), Tensor(np.random.rand(2, 3, 4).astype(np.float32))),
        (Tensor(np.random.rand(2, 3, 1).astype(np.float32)), Tensor(np.random.rand(2, 3, 4).astype(np.float32))),
        (Tensor(np.random.rand(1, 3, 4).astype(np.float32)), Tensor(np.random.rand(2, 3, 4).astype(np.float32))),
        (Tensor(np.random.rand(1, 3, 4).astype(np.float32)), Tensor(np.random.rand(1, 3, 4).astype(np.float32))),
    ]
    for x, y in input_pairs:
        pynative_net = MultiplySquareNet()
        pynative_net.set_grad()
        pynative_output = pynative_net(x, y)
        sens = Tensor(np.random.randn(*pynative_output.shape).astype(np.float32))
        pynative_grads = _compute_grad_of_net_inputs(pynative_net, x, y)

        jit_net = MultiplySquareNet()
        jit_net.set_grad()
        jit_net = _apply_jit(jit_net)
        jit_output = jit_net(x, y)
        jit_grads = _compute_grad_of_net_inputs(jit_net, x, y)

        match_array(jit_output, pynative_output)
        assert_equal(jit_grads, pynative_grads, decimal=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_shape_tuple_tensor_input():
    """
    Feature: Dynamic shape with tuple tensor inputs.
    Description: Change tensor shapes inside a tuple argument across executions.
    Expectation: JIT forward outputs and gradients match pynative for every tuple.
    Migrated from: test_dynamic_shape_pijit.py::test_dynamic_shape_pijit_change_tuple_tensor_shape
    """
    tensor_pairs = [
        [Tensor(np.random.rand(2, 3, 4).astype(np.float32)), Tensor(np.random.rand(2, 3, 4).astype(np.float32))],
        [Tensor(np.random.rand(3, 3, 4).astype(np.float32)), Tensor(np.random.rand(3, 3, 4).astype(np.float32))],
        [Tensor(np.random.rand(4, 3, 4).astype(np.float32)), Tensor(np.random.rand(4, 3, 4).astype(np.float32))],
    ]
    for tensors in tensor_pairs:
        pynative_net = TupleSelectorNet()
        pynative_net.set_grad()
        pynative_output = pynative_net(tensors)
        sens = Tensor(np.random.randn(*pynative_output.shape).astype(np.float32))
        pynative_grad = _compute_grad_of_net_inputs(pynative_net, tensors)

        jit_net = TupleSelectorNet()
        jit_net.set_grad()
        jit_net = _apply_jit(jit_net)
        jit_output = jit_net(tensors)
        jit_grad = _compute_grad_of_net_inputs(jit_net, tensors)

        match_array(jit_output, pynative_output)
        assert_equal(jit_grad, pynative_grad, decimal=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_shape_parameter_input():
    """
    Feature: Dynamic shape with Parameter input.
    Description: Change Parameter shape across executions while using in-place assign_add.
    Expectation: JIT forward outputs and gradients match pynative for every shape.
    Migrated from: test_dynamic_shape_pijit.py::test_dynamic_shape_pijit_change_parameter_shape
    """
    shapes = [(2, 3, 4), (3, 3, 4), (4, 3, 4)]
    for idx, shape in enumerate(shapes):
        param_data = np.random.rand(*shape).astype(np.float32)
        y_data = np.random.rand(*shape).astype(np.float32)

        pynative_x = Parameter(Tensor(param_data.copy()), name=f"x_param_{idx}_pn")
        pynative_y = Tensor(y_data.copy())

        pynative_net = AssignAddNet()
        pynative_net.set_grad()
        pynative_output = pynative_net(pynative_x, pynative_y)
        sens = Tensor(np.random.randn(*pynative_output.shape).astype(np.float32))
        pynative_grads = _compute_grad_of_net_inputs(pynative_net, pynative_x, pynative_y)

        jit_x = Parameter(Tensor(param_data.copy()), name=f"x_param_{idx}_jit")
        jit_y = Tensor(y_data.copy())

        jit_net = AssignAddNet()
        jit_net.set_grad()
        jit_net = _apply_jit(jit_net, jit_config={"limit_graph_count": 2})
        jit_output = jit_net(jit_x, jit_y)
        jit_grads = _compute_grad_of_net_inputs(jit_net, jit_x, jit_y)

        match_array(jit_output, pynative_output)
        assert_equal(jit_grads, pynative_grads, decimal=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_shape_tensor_with_none():
    """
    Feature: Dynamic shape with optional argument.
    Description: Change input tensor shape while passing None for the second argument.
    Expectation: JIT forward outputs and gradients match pynative for every shape.
    Migrated from: test_dynamic_shape_pijit.py::test_dynamic_shape_pijit_change_shape_with_none
    """
    tensors = [
        Tensor(np.random.rand(2, 2).astype(np.float32)),
        Tensor(np.random.rand(3, 3).astype(np.float32)),
        Tensor(np.random.rand(6, 6).astype(np.float32)),
    ]
    axis = None
    for x in tensors:
        pynative_net = EyeNet()
        pynative_net.set_grad()
        pynative_output = pynative_net(x, axis)
        sens = Tensor(np.random.randn(*pynative_output.shape).astype(np.float32))
        pynative_grads = _compute_grad_of_net_inputs(pynative_net, x, axis)

        jit_net = EyeNet()
        jit_net.set_grad()
        jit_net = _apply_jit(jit_net)
        jit_output = jit_net(x, axis)
        jit_grads = _compute_grad_of_net_inputs(jit_net, x, axis)

        match_array(jit_output, pynative_output)
        assert_equal(jit_grads, pynative_grads, decimal=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_shape_tuple_int_input():
    """
    Feature: Dynamic shape with tuple integer inputs.
    Description: Change the tuple length across executions when creating tensors from shape tuples.
    Expectation: JIT forward outputs match pynative for every shape.
    Migrated from: test_dynamic_shape_pijit.py::test_dynamic_shape_pijit_change_tuple_int
    """
    shapes = [(2, 3), (2, 3, 4), (2, 3, 4, 5)]
    for shape in shapes:
        pynative_net = OnesNet()
        pynative_output = pynative_net(shape)

        jit_net = OnesNet()
        jit_net = _apply_jit(jit_net)
        jit_output = jit_net(shape)

        match_array(jit_output, pynative_output)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_shape_tensor_and_scalar_input():
    """
    Feature: Dynamic shape with tensor-scalar inputs.
    Description: Change tensor shapes and scalar values across executions.
    Expectation: JIT forward outputs and gradients match pynative for every input pair.
    Migrated from: test_dynamic_shape_pijit.py::test_dynamic_shape_pijit_change_shape_and_float
    """
    cases = [
        (np.random.rand(2, 3, 4).astype(np.float32), 1.0),
        (np.random.rand(2, 3, 4).astype(np.float32), 2.0),
        (np.random.rand(2, 3, 6).astype(np.float32), 2.0),
    ]
    for data, scalar in cases:
        x = Tensor(data)

        pynative_net = TensorScalarNet()
        pynative_net.set_grad()
        pynative_output = pynative_net(x, scalar)
        pynative_grads = _compute_grad_of_net_inputs(pynative_net, x, scalar)

        jit_net = TensorScalarNet()
        jit_net.set_grad()
        jit_net = _apply_jit(jit_net)
        jit_output = jit_net(x, scalar)
        jit_grads = _compute_grad_of_net_inputs(jit_net, x, scalar)

        match_array(jit_output, pynative_output)
        assert_equal(jit_grads, pynative_grads, decimal=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_shape_multi_output_variation():
    """
    Feature: Dynamic shape with tuple outputs.
    Description: Change input shapes and ranks for a cell returning multiple tensors.
    Expectation: JIT forward outputs match pynative for every input pair.
    Migrated from: test_dynamic_shape_pijit.py::test_dynamic_shape_pijit_change_shape_and_rank
    """
    input_pairs = [
        (Tensor(np.random.rand(1, 3).astype(np.float32)), Tensor(np.random.rand(1, 2, 2).astype(np.float32))),
        (Tensor(np.random.rand(2, 1).astype(np.float32)), Tensor(np.random.rand(4).astype(np.float32))),
        (Tensor(np.random.rand(1, 3).astype(np.float32)), Tensor(np.random.rand(4).astype(np.float32))),
    ]
    for x, y in input_pairs:
        pynative_net = MultiOutputNet()
        pynative_output = pynative_net(x, y)

        jit_net = MultiOutputNet()
        jit_net = _apply_jit(jit_net)
        jit_output = jit_net(x, y)

        assert_equal(jit_output, pynative_output)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_shape_multi_output_change_shape():
    """
    Feature: Dynamic shape with tuple outputs.
    Description: Run the multi-output cell multiple times while changing input shapes and ranks.
    Expectation: JIT forward outputs match pynative and dynamic compilation happens for every shape.
    Migrated from: test_dynamic_shape_pijit.py::test_dynamic_shape_pijit_change_shape
    """
    cases = [
        (Tensor(np.random.rand(1, 3).astype(np.float32)), Tensor(np.random.rand(1, 2, 2).astype(np.float32))),
        (Tensor(np.random.rand(2, 1).astype(np.float32)), Tensor(np.random.rand(4).astype(np.float32))),
        (Tensor(np.random.rand(1, 3).astype(np.float32)), Tensor(np.random.rand(4).astype(np.float32))),
    ]

    pynative_net = MultiOutputNet()
    expected_results = [pynative_net(x, y) for x, y in cases]

    jit_net = MultiOutputNet()
    jit_net = _apply_jit(jit_net, jit_config={"limit_graph_count": 1})

    for (x, y), expected in zip(cases, expected_results):
        actual = jit_net(x, y)
        assert_equal(actual, expected)
