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
import mindspore.ops as ops
from mindspore import context, Tensor
from mindspore.scipy.ops import Eigh, Cholesky, SolveTriangular
from tests.st.scipy_st.utils import create_random_rank_matrix, create_sym_pos_matrix, gradient_check


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('shape', [(8, 8)])
@pytest.mark.parametrize('data_type', [(onp.float32, 1e-2, 1e-3), (onp.float64, 1e-4, 1e-7)])
def test_cholesky_grad(shape, data_type):
    """
    Feature: ALL TO ALL
    Description: test cases for grad implementation of cholesky operator in graph mode and pynative mode.
    Expectation: the result match gradient checking.
    """
    onp.random.seed(0)
    context.set_context(mode=context.GRAPH_MODE)
    dtype, epsilon, error = data_type

    class CholeskyNet(nn.Cell):
        def __init__(self):
            super(CholeskyNet, self).__init__()
            self.mean = ops.ReduceMean()
            # Input arg clean not supports grad right now, just default clean to True.
            self.cholesky = Cholesky(clean=True)

        def construct(self, a):
            c = self.cholesky(a)
            return self.mean(c)

    cholesky_net = CholeskyNet()
    a = create_sym_pos_matrix(shape, dtype)
    cholesky_net(Tensor(a))
    assert gradient_check(Tensor(a), cholesky_net, epsilon) < error
    context.set_context(mode=context.PYNATIVE_MODE)
    cholesky_net(Tensor(a))
    assert gradient_check(Tensor(a), cholesky_net, epsilon) < error


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
    context.set_context(mode=context.GRAPH_MODE)
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
    assert gradient_check(Tensor(a), net, epsilon) < error


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('shapes', [((8, 8), (8, 8)), ((8, 8), (8, 2)), ((8, 8), (8,))])
@pytest.mark.parametrize('trans', ["N", "T"])
@pytest.mark.parametrize('lower', [False, True])
@pytest.mark.parametrize('unit_diagonal', [True, False])
@pytest.mark.parametrize('data_type', [(onp.float32, 1e-3, 1e-3), (onp.float64, 1e-4, 1e-7)])
def test_trsm_grad(shapes, trans, lower, unit_diagonal, data_type):
    """
    Feature: ALL TO ALL
    Description: test cases for grad implementation of SolveTriangular operator
    Expectation: the result match gradient checking.
    """
    a_shape, b_shape = shapes
    onp.random.seed(0)
    context.set_context(mode=context.GRAPH_MODE)
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
    assert gradient_check([Tensor(a), Tensor(b)], net, epsilon) < error
