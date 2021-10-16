# Copyright 2021 Huawei Technologies Co., Ltd
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
from typing import Generic
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import PrimitiveWithInfer
from mindspore.ops import prim_attr_register
from scipy.linalg import lu_factor
from scipy.linalg import lu_solve
import numpy as np
import pytest

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class LU(PrimitiveWithInfer):
    """
    LU decomposition with partial pivoting
    P.A = L.U
    """

    @prim_attr_register
    def __init__(self):
        super().__init__(name="LU")
        self.init_prim_io_names(inputs=['x'], outputs=['lu', 'pivots', 'permutation'])

    def __infer__(self, x):
        x_shape = list(x['shape'])
        x_dtype = x['dtype']
        pivots_shape = []
        permutation_shape = []
        ndim = len(x_shape)
        if ndim == 0:
            pivots_shape = x_shape
            permutation_shape = x_shape
        elif ndim == 1:
            pivots_shape = x_shape[:-1]
            permutation_shape = x_shape[:-1]
        else:
            pivots_shape = x_shape[-2:-1]
            permutation_shape = x_shape[-2:-1]

        output = {
            'shape': (x_shape, pivots_shape, permutation_shape),
            'dtype': (x_dtype, mstype.int32, mstype.int32),
            'value': None
        }
        return output


class LUSolver(PrimitiveWithInfer):
    """
    LUSolver for Ax = b
    """

    @prim_attr_register
    def __init__(self):
        super().__init__(name="LUSolver")
        self.init_prim_io_names(inputs=['x', 'b'], outputs=['output'])

    def __infer__(self, x, b):
        b_shape = list(b['shape'])
        x_dtype = x['dtype']
        output = {
            'shape': tuple(b_shape),
            'dtype': x_dtype,
            'value': None
        }
        return output


class LuNet(nn.Cell):
    def __init__(self):
        super(LuNet, self).__init__()
        self.lu = LU()

    def construct(self, a):
        return self.lu(a)


class LUSolverNet(nn.Cell):
    def __init__(self):
        super(LUSolverNet, self).__init__()
        self.lu_solver = LUSolver()

    def construct(self, a, b):
        return self.lu_solver(a, b)


def _match_array(actual, expected, error=0):
    if isinstance(actual, int):
        actual = np.asarray(actual)
    if isinstance(actual, tuple):
        actual = np.asarray(actual)

    if error > 0:
        np.testing.assert_almost_equal(actual, expected, decimal=error)
    else:
        np.testing.assert_equal(actual, expected)


@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('n', [10, 20])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_lu_net(n: int, dtype: Generic):
    """
    Feature: ALL To ALL
    Description: test cases for lu decomposition test cases for A[N,N]x = b[N,1]
    Expectation: the result match to scipy
    """
    a = (np.random.random((n, n)) + np.eye(n)).astype(dtype)
    s_lu, _ = lu_factor(a)
    mscp_lu_net = LuNet()
    tensor_a = Tensor(a)
    mscp_lu, _, _ = mscp_lu_net(tensor_a)
    _match_array(mscp_lu.asnumpy(), s_lu, error=4)


@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('n', [10, 20])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_lu_solver_net(n: int, dtype: Generic):
    """
    Feature: ALL To ALL
    Description: test cases for lu_solve test cases for A[N,N]x = b[N,1]
    Expectation: the result match to scipy
    """
    a = (np.random.random((n, n)) + np.eye(n)).astype(dtype)
    b = np.random.random((n, 1)).astype(dtype)
    s_lu, s_piv = lu_factor(a)
    lu_factor_x = (s_lu, s_piv)
    scp_x = lu_solve(lu_factor_x, b)
    mscp_lu_net = LUSolverNet()
    tensor_a = Tensor(a)
    tensor_b = Tensor(b)
    mscp_x = mscp_lu_net(tensor_a, tensor_b)
    _match_array(mscp_x.asnumpy(), scp_x, error=4)
