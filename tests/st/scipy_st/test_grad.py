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
"""st for scipy.ops_grad."""
import pytest
import numpy as onp
import mindspore.nn as nn
from mindspore.nn import Cell
import mindspore.ops as ops
from mindspore.ops import composite as C
from mindspore import context, Tensor
from mindspore.scipy.linalg import cho_factor, cho_solve
from mindspore.ops.operations.math_ops import Cholesky
from mindspore.scipy.ops import Eigh, SolveTriangular
from tests.st.scipy_st.utils import create_random_rank_matrix, create_sym_pos_matrix, gradient_check


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cholesky_grad():
    """
    Feature: ALL TO ALL
    Description: test cases for grad implementation of cholesky operator in graph mode and pynative mode.
    Expectation: the result match gradient checking.
    """
    context.set_context(mode=context.GRAPH_MODE)
    class CholeskyNet(nn.Cell):
        def __init__(self):
            super(CholeskyNet, self).__init__()
            # Input arg clean not supports grad right now, just default clean to True.
            self.cholesky = Cholesky()

        def construct(self, a):
            return self.cholesky(a)

    class CholeskyGradNet(Cell):
        def __init__(self, network):
            super(CholeskyGradNet, self).__init__()
            self.grad = C.GradOperation(get_all=True, sens_param=True)
            self.network = network

        def construct(self, input_data, grad_np):
            gout = self.grad(self.network)(input_data, grad_np)
            return gout

    x_np = onp.array([[10, 22], [22, 50]]).astype(onp.float32)
    net = CholeskyNet()
    output_ms = net(Tensor(x_np))
    grad_np = onp.array([[1, 0], [0, 1]]).astype(onp.float32)
    grad_net = CholeskyGradNet(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(grad_np))
    expect_output = onp.array([[3.1622777, 0], [6.9570107, 1.2649117]])
    expect_grad_output = onp.array([[2.071291, -0.869626], [-0.869626, 0.39528453]])
    assert onp.allclose(output_ms.asnumpy(), expect_output)
    assert onp.allclose(output_grad_ms[0].asnumpy(), expect_grad_output)

    context.set_context(mode=context.PYNATIVE_MODE)
    x_np = onp.array([[12.56, 27.28], [27.28, 60.5]]).astype(onp.float64)
    net = CholeskyNet()
    output_ms = net(Tensor(x_np))
    grad_np = onp.array([[1, 0], [0, 1]]).astype(onp.float64)
    grad_net = CholeskyGradNet(net)
    output_grad_ms = grad_net(Tensor(x_np), Tensor(grad_np))
    expect_output = onp.array([[3.544009, 0.], [7.697497, 1.1173816]])
    expect_grad_output = onp.array([[2.252033, -0.9719036], [-0.9719037, 0.44747472]])
    assert onp.allclose(output_ms.asnumpy(), expect_output)
    assert onp.allclose(output_grad_ms[0].asnumpy(), expect_grad_output)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cho_factor_grad():
    """
    Feature: ALL TO ALL
    Description: test cases for grad implementation of cho_factor in graph mode and pynative mode.
    Expectation: the result match gradient checking.
    """
    context.set_context(mode=context.GRAPH_MODE)

    class ChoFactorNet(nn.Cell):
        def __init__(self, lower):
            super(ChoFactorNet, self).__init__()
            self.lower = lower

        def construct(self, a):
            c, _ = cho_factor(a, self.lower)
            return c

    a = Tensor(onp.array([[3, 1], [1, 4]]).astype(onp.float32))
    c, _ = cho_factor(a, True)
    expect_output = onp.array([[1.7320508, 0.], [0.57735026, 1.9148543]])
    assert onp.allclose(c.asnumpy(), expect_output)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('lower', [True, False])
@pytest.mark.parametrize('shape', [(8, 8)])
@pytest.mark.parametrize('data_type', [(onp.float32, 1e-2, 1e-3), (onp.float64, 1e-4, 1e-7)])
def test_cho_solve_grad(lower, shape, data_type):
    """
    Feature: ALL TO ALL
    Description: test cases for grad implementation of cho_solve in graph mode and pynative mode.
    Expectation: the result match gradient checking.
    """
    onp.random.seed(0)
    context.set_context(mode=context.GRAPH_MODE)
    dtype, epsilon, error = data_type

    class ChoSolveNet(nn.Cell):
        def __init__(self, lower):
            super(ChoSolveNet, self).__init__()
            self.mean = ops.ReduceMean()
            self.lower = lower

        def construct(self, c, b):
            c_lower = (c, self.lower)
            output = cho_solve(c_lower, b)
            return self.mean(output)

    a = create_sym_pos_matrix(shape, dtype)
    n = shape[-1]
    b = onp.ones((n, 1), dtype=dtype)
    msp_c, msp_lower = cho_factor(Tensor(a), lower)
    cho_solve_net = ChoSolveNet(msp_lower)
    assert gradient_check([msp_c, Tensor(b)], cho_solve_net, epsilon) < error
    context.set_context(mode=context.PYNATIVE_MODE)
    assert gradient_check([msp_c, Tensor(b)], cho_solve_net, epsilon) < error


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('compute_eigenvectors', [True, False])
@pytest.mark.parametrize('lower', [True, False])
@pytest.mark.parametrize('shape', [(8, 8)])
@pytest.mark.parametrize('data_type', [(onp.float32, 1e-3, 1e-3), (onp.float64, 1e-4, 1e-7)])
def test_eigh_grad(compute_eigenvectors, lower, shape, data_type):
    """
    Feature: ALL TO ALL
    Description: test cases for grad implementation of Eigh operator
    Expectation: the result match gradient checking.
    """
    onp.random.seed(0)
    dtype, epsilon, error = data_type

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.mean = ops.ReduceMean()
            self.sum = ops.ReduceSum()
            self.compute_eigenvectors = compute_eigenvectors
            self.lower = lower
            self.eigh = Eigh(compute_eigenvectors, lower)

        def construct(self, a):
            res = None
            if self.compute_eigenvectors:
                w, v = self.eigh(a)
                res = self.sum(w) + self.mean(v)
            else:
                w = self.eigh(a)
                res = self.mean(w)
            return res

    net = Net()
    a = create_random_rank_matrix(shape, dtype)
    context.set_context(mode=context.GRAPH_MODE)
    assert gradient_check(Tensor(a), net, epsilon) < error
    context.set_context(mode=context.PYNATIVE_MODE)
    assert gradient_check(Tensor(a), net, epsilon) < error


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('shapes', [((8, 8), (8, 8)), ((8, 8), (8, 2)), ((8, 8), (8,))])
@pytest.mark.parametrize('trans', ["N", "T", "C"])
@pytest.mark.parametrize('lower', [False, True])
@pytest.mark.parametrize('unit_diagonal', [True, False])
@pytest.mark.parametrize('data_type', [(onp.float32, 1e-3, 1e-3), (onp.float64, 1e-4, 1e-7)])
def test_trsm_grad_pynative(shapes, trans, lower, unit_diagonal, data_type):
    """
    Feature: ALL TO ALL
    Description: test cases for grad implementation of SolveTriangular operator in PYNATIVE mode.
    Expectation: the result match gradient checking.
    """
    a_shape, b_shape = shapes
    onp.random.seed(0)
    dtype, epsilon, error = data_type

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.mean = ops.ReduceMean()
            self.sum = ops.ReduceSum()
            self.trsm = SolveTriangular(lower, unit_diagonal, trans)

        def construct(self, a, b):
            x = self.trsm(a, b)
            return self.sum(x) + self.mean(x)

    net = Net()
    a = (onp.random.random(a_shape) + onp.eye(a_shape[-1])).astype(dtype)
    b = onp.random.random(b_shape).astype(dtype)
    context.set_context(mode=context.PYNATIVE_MODE)
    assert gradient_check([Tensor(a), Tensor(b)], net, epsilon) < error


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('shapes', [((8, 8), (8, 8)), ((8, 8), (8, 2)), ((8, 8), (8,))])
@pytest.mark.parametrize('trans', ["N", "T", "C"])
@pytest.mark.parametrize('lower', [False, True])
@pytest.mark.parametrize('unit_diagonal', [True, False])
@pytest.mark.parametrize('data_type', [(onp.float32, 1e-3, 1e-3), (onp.float64, 1e-4, 1e-7)])
def test_trsm_grad_graph(shapes, trans, lower, unit_diagonal, data_type):
    """
    Feature: ALL TO ALL
    Description: test cases for grad implementation of SolveTriangular operator in GRAPH mode.
    Expectation: the result match gradient checking.
    """
    a_shape, b_shape = shapes
    onp.random.seed(0)
    dtype, epsilon, error = data_type

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.mean = ops.ReduceMean()
            self.sum = ops.ReduceSum()
            self.trsm = SolveTriangular(lower, unit_diagonal, trans)

        def construct(self, a, b):
            x = self.trsm(a, b)
            return self.sum(x) + self.mean(x)

    net = Net()
    a = (onp.random.random(a_shape) + onp.eye(a_shape[-1])).astype(dtype)
    b = onp.random.random(b_shape).astype(dtype)
    context.set_context(mode=context.GRAPH_MODE)
    assert gradient_check([Tensor(a), Tensor(b)], net, epsilon) < error
