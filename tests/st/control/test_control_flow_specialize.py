# Copyright 2022 Huawei Technologies Co., Ltd
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
""" test_control_flow_specialize """
from mindspore.nn import Cell
from mindspore.common import Tensor, dtype, Parameter
from mindspore.ops import operations as P
from mindspore import ms_function
import mindspore.ops.functional as F
import numpy as np
import pytest


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_renormalization_after_cconv_poly_node():
    """
    Feature: control flow
    Description: In the renormalization after cconv, there should be no poly node error.
    Expectation: No exception.
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor([(- 1)], dtype.float32), name='w')
            self.b = Parameter(Tensor([(- 1)], dtype.float32), name='b')

        def construct(self, x, y):
            def inner(x):
                if x >= 5:
                    return x
                return x

            def outer(x):
                if x >= inner(x):
                    return x
                return x

            while self.b == 0:
                if outer(self.b) <= self.b:
                    y = self.w + outer(self.w)
                if y > inner(self.b):
                    break
            return x + y

    x = np.array([5], np.float32)
    y = np.array([3], np.float32)
    net1 = Net()
    grad_net = F.grad(net1, grad_position=(0, 1))
    expected = np.array([1], np.float32)
    output = grad_net(Tensor(x), Tensor(y))
    assert np.allclose(expected, output[0].asnumpy(), 0.0001)
    assert np.allclose(expected, output[1].asnumpy(), 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_poly_delay_specialize():
    """
    Feature: Specialize.
    Description: If a poly node's parent are not specialized, poly node should be delay specialized.
    Expectation: graph can be executed and no exception raised.
    """
    pow_ops = P.Pow()

    @ms_function
    def poly_node_network(x, y):
        def function_h():
            pow_res = pow_ops(x, x)

            def function_g(param_x):
                return pow(pow_res, param_x)

            return F.make_tuple(pow_res, function_g)

        def function_f():
            h_out = function_h()
            h_forward_out = F.tuple_getitem(h_out, 0)
            g = F.tuple_getitem(h_out, 1)

            def function_k():
                kout1 = g(x)
                kout2 = g(y)
                kout = F.depend(kout1, kout2)
                return kout

            return F.make_tuple(h_forward_out, function_k)

        out = function_f()
        forward_out = F.tuple_getitem(out, 0)
        closure_out = F.tuple_getitem(out, 1)
        closure_out_tensor = closure_out()
        return F.add(forward_out, closure_out_tensor)

    x = Tensor([1], dtype.int32)
    y = Tensor([1, 2], dtype.int32)
    poly_node_network(x, y)


def test_renormalization_cannot_find_specialized_abstract():
    """
    Feature: control flow
    Description: after renormalization, funcgraph with different args in different abstracts are broadened
                 to use the same funcgraph, so all these abstracts have to map the same abstract, then
                 the backend can find that specialized func_graph.
    Expectation: No exception.
    """
    def foo(x, y):
        for e in range(2):
            x = e * y
            if x >= 2:
                break

        if y >= x:
            x = x * y

        return x + y

    x = np.array([5], np.int32)
    y = np.array([3], np.int32)
    grad_foo = F.grad(foo, grad_position=(0, 1))
    output = grad_foo(Tensor(x), Tensor(y))
    assert output[0].asnumpy() == np.array([0], np.int32)
    assert output[1].asnumpy() == np.array([7], np.int32)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_renormalization_cannot_find_specialized_abstract_2():
    """
    Feature: control flow
    Description: after renormalization, funcgraph with different args in different abstracts are broadened
                 to use the same funcgraph, so all these abstracts have to map the same abstract, then
                 the backend can find that specialized func_graph.
    Expectation: No exception.
    """
    def foo(x, y):
        for e in range(2):
            x = e * y
            if x >= 2:
                break

        if y >= 4:
            x = x * y
        elif y >= x:
            x = x * y

        return x + y

    x = np.array([5], np.int32)
    y = np.array([3], np.int32)
    grad_foo = F.grad(foo, grad_position=(0, 1))
    output = grad_foo(Tensor(x), Tensor(y))
    assert output[0].asnumpy() == np.array([0], np.int32)
    assert output[1].asnumpy() == np.array([7], np.int32)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_renormalization_cannot_find_specialized_abstract_2nd_grad():
    """
    Feature: control flow
    Description: after renormalization, funcgraph with different args in different abstracts are broadened
                 to use the same funcgraph, so all these abstracts have to map the same abstract, then
                 the backend can find that specialized func_graph.
    Expectation: No exception.
    """
    def foo(x):
        out = x
        for e in range(2):
            x = e * out
            if x >= 2:
                break

        if out >= x:
            out = x * out

        return x + out

    x = np.array([5], np.int32)
    grad_foo = F.grad(foo, grad_position=(0,))
    grad_foo_2nd = F.grad(grad_foo, grad_position=(0,))
    output = grad_foo_2nd(Tensor(x))
    assert output[0].asnumpy() == np.array([2], np.int32)
