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
"""test for solve eigenvalues & eigen vectors"""

import pytest
import numpy as np
import mindspore as msp
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops import PrimitiveWithInfer, prim_attr_register
from mindspore._checkparam import Validator as validator

np.random.seed(0)


class Eig(PrimitiveWithInfer):
    """
    Eig decomposition,(generic matrix)
    Ax = lambda * x
    """

    @prim_attr_register
    def __init__(self, compute_eigenvectors):
        super().__init__(name="Eig")
        self.init_prim_io_names(inputs=['A'], outputs=['output', 'output_v'])
        self.compute_eigenvectors = validator.check_value_type(
            "compute_eigenvectors", compute_eigenvectors, [bool], self.name)

    def __infer__(self, A):
        shape = {}
        if A['dtype'] == msp.tensor_type(msp.dtype.float32):
            shape = {
                'shape': ((A['shape'][0],), (A['shape'][0], A['shape'][0])),
                'dtype': (msp.complex64, msp.complex64),
                'value': None
            }
        elif A['dtype'] == msp.tensor_type(msp.dtype.float64):
            shape = {
                'shape': ((A['shape'][0],), (A['shape'][0], A['shape'][0])),
                'dtype': (msp.complex128, msp.complex128),
                'value': None
            }
        elif A['dtype'] == msp.tensor_type(msp.dtype.complex64):
            shape = {
                'shape': ((A['shape'][0],), (A['shape'][0], A['shape'][0])),
                'dtype': (msp.complex64, msp.complex64),
                'value': None
            }
        elif A['dtype'] == msp.tensor_type(msp.dtype.complex128):
            shape = {
                'shape': ((A['shape'][0],), (A['shape'][0], A['shape'][0])),
                'dtype': (msp.complex128, msp.complex128),
                'value': None
            }
        return shape


class EigNet(nn.Cell):
    def __init__(self, b):
        super(EigNet, self).__init__()
        self.b = b
        self.eig = Eig(b)

    def construct(self, A):
        r = self.eig(A)
        if self.b:
            return (r[0], r[1])
        return (r[0],)


def match(v, v_, error=0):
    if error > 0:
        np.testing.assert_almost_equal(v, v_, decimal=error)
    else:
        np.testing.assert_equal(v, v_)


def create_sym_pos_matrix(m, n, dtype):
    a = (np.random.random((m, n)) + np.eye(m, n)).astype(dtype)
    return np.dot(a, a.T)


@pytest.mark.parametrize('n', [4, 6, 9, 10])
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.platform_x86_cpu
def test_eig_net(n: int, mode):
    """
    Feature: ALL To ALL
    Description: test cases for eigen decomposition test cases for Ax= lambda * x /( A- lambda * E)X=0
    Expectation: the result match to numpy
    """
    # test for real scalar float 32
    rtol = 1e-4
    atol = 1e-5
    msp_eig = EigNet(True)
    A = create_sym_pos_matrix(n, n, np.float32)
    tensor_a = Tensor(np.array(A).astype(np.float32))
    msp_w, msp_v = msp_eig(tensor_a)
    assert np.allclose(A @ msp_v.asnumpy() - msp_v.asnumpy() @ np.diag(msp_w.asnumpy()), np.zeros((n, n)), rtol, atol)

    # test case for real scalar double 64
    A = np.random.rand(n, n)
    rtol = 1e-5
    atol = 1e-8
    msp_eig = EigNet(True)
    msp_w, msp_v = msp_eig(Tensor(np.array(A).astype(np.float64)))

    # Compare with scipy
    # sp_w, sp_v = sp.linalg.eig(A.astype(np.float64))
    assert np.allclose(A @ msp_v.asnumpy() - msp_v.asnumpy() @ np.diag(msp_w.asnumpy()), np.zeros((n, n)), rtol, atol)

    # test case for complex64
    rtol = 1e-4
    atol = 1e-5
    A = np.array(np.random.rand(n, n), dtype=np.complex64)
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                A[i][j] = complex(np.random.rand(1, 1), 0)
            else:
                A[i][j] = complex(np.random.rand(1, 1), np.random.rand(1, 1))
    msp_eig = EigNet(True)
    msp_w, msp_v = msp_eig(Tensor(np.array(A).astype(np.complex64)))
    # Compare with scipy, scipy passed
    # sp_w, sp_v = sp.linalg.eig(A.astype(np.complex128))
    # assert np.allclose(A @ sp_v - sp_v @ np.diag(sp_w), np.zeros((n, n)), rtol, atol)

    # print(A @ msp_v.asnumpy() - msp_v.asnumpy() @ np.diag(msp_w.asnumpy()))
    assert np.allclose(A @ msp_v.asnumpy() - msp_v.asnumpy() @ np.diag(msp_w.asnumpy()), np.zeros((n, n)), rtol, atol)

    # test for complex128
    rtol = 1e-5
    atol = 1e-8
    A = np.array(np.random.rand(n, n), dtype=np.complex128)
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                A[i][j] = complex(np.random.rand(1, 1), 0)
            else:
                A[i][j] = complex(np.random.rand(1, 1), np.random.rand(1, 1))
    msp_eig = EigNet(True)
    msp_w, msp_v = msp_eig(Tensor(np.array(A).astype(np.complex128)))
    # Compare with scipy, scipy passed
    # sp_w, sp_v = sp.linalg.eig(A.astype(np.complex128))
    # assert np.allclose(A @ sp_v - sp_v @ np.diag(sp_w), np.zeros((n, n)), rtol, atol)

    # print(A @ msp_v.asnumpy() - msp_v.asnumpy() @ np.diag(msp_w.asnumpy()))
    assert np.allclose(A @ msp_v.asnumpy() - msp_v.asnumpy() @ np.diag(msp_w.asnumpy()), np.zeros((n, n)), rtol, atol)
