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
from mindspore import Tensor
from mindspore.scipy.ops import EighNet

np.random.seed(0)


def match(v, v_, error=0):
    if error > 0:
        np.testing.assert_almost_equal(v, v_, decimal=error)
    else:
        np.testing.assert_equal(v, v_)


def create_sym_pos_matrix(m, n, dtype):
    a = (np.random.random((m, n)) + np.eye(m, n)).astype(dtype)
    return np.dot(a, a.T)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [4, 6, 8, 20])
def test_eigh_net(n: int):
    """
    Feature: ALL To ALL
    Description: test cases for eigen decomposition test cases for Ax= lambda * x /( A- lambda * E)X=0
    Expectation: the result match to numpy
    """
    # test for real scalar float 32
    rtol = 1e-3
    atol = 1e-4
    A = create_sym_pos_matrix(n, n, np.float32)
    msp_eigh = EighNet(True, True)
    msp_wl, msp_vl = msp_eigh(Tensor(np.array(A).astype(np.float32)))
    msp_eigh = EighNet(True, False)
    msp_wu, msp_vu = msp_eigh(Tensor(np.array(A).astype(np.float32)))
    assert np.allclose(A @ msp_vl.asnumpy() - msp_vl.asnumpy() @ np.diag(msp_wl.asnumpy()), np.zeros((n, n)), rtol,
                       atol)
    assert np.allclose(A @ msp_vu.asnumpy() - msp_vu.asnumpy() @ np.diag(msp_wu.asnumpy()), np.zeros((n, n)), rtol,
                       atol)

    # test case for real scalar double 64
    A = create_sym_pos_matrix(n, n, np.float64)
    rtol = 1e-5
    atol = 1e-8
    msp_eigh = EighNet(True, True)
    msp_wl, msp_vl = msp_eigh(Tensor(np.array(A).astype(np.float64)))
    msp_eigh = EighNet(True, False)
    msp_wu, msp_vu = msp_eigh(Tensor(np.array(A).astype(np.float64)))
    assert np.allclose(A @ msp_vl.asnumpy() - msp_vl.asnumpy() @ np.diag(msp_wl.asnumpy()), np.zeros((n, n)), rtol,
                       atol)
    assert np.allclose(A @ msp_vu.asnumpy() - msp_vu.asnumpy() @ np.diag(msp_wu.asnumpy()), np.zeros((n, n)), rtol,
                       atol)
    # test for real scalar float64 no vector
    msp_eigh = EighNet(False, True)
    msp_wl0 = msp_eigh(Tensor(np.array(A).astype(np.float64)))
    msp_eigh = EighNet(False, False)
    msp_wu0 = msp_eigh(Tensor(np.array(A).astype(np.float64)))
    assert np.allclose(msp_wl.asnumpy() - msp_wl0.asnumpy(), np.zeros((n, n)), rtol, atol)
    assert np.allclose(msp_wu.asnumpy() - msp_wu0.asnumpy(), np.zeros((n, n)), rtol, atol)

    # test case for complex64
    rtol = 1e-3
    atol = 1e-4
    A = np.array(np.random.rand(n, n), dtype=np.complex64)
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                A[i][j] = complex(np.random.rand(1, 1), 0)
            else:
                A[i][j] = complex(np.random.rand(1, 1), np.random.rand(1, 1))
    sym_Al = (np.tril((np.tril(A) - np.tril(A).T)) + np.tril(A).conj().T)
    sym_Au = (np.triu((np.triu(A) - np.triu(A).T)) + np.triu(A).conj().T)
    msp_eigh = EighNet(True, True)
    msp_wl, msp_vl = msp_eigh(Tensor(np.array(A).astype(np.complex64)))
    msp_eigh = EighNet(True, False)
    msp_wu, msp_vu = msp_eigh(Tensor(np.array(A).astype(np.complex64)))
    assert np.allclose(sym_Al @ msp_vl.asnumpy() - msp_vl.asnumpy() @ np.diag(msp_wl.asnumpy()),
                       np.zeros((n, n)), rtol, atol)
    assert np.allclose(sym_Au @ msp_vu.asnumpy() - msp_vu.asnumpy() @ np.diag(msp_wu.asnumpy()),
                       np.zeros((n, n)), rtol, atol)

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
    sym_Al = (np.tril((np.tril(A) - np.tril(A).T)) + np.tril(A).conj().T)
    sym_Au = (np.triu((np.triu(A) - np.triu(A).T)) + np.triu(A).conj().T)
    msp_eigh = EighNet(True, True)
    msp_wl, msp_vl = msp_eigh(Tensor(np.array(A).astype(np.complex128)))
    msp_eigh = EighNet(True, False)
    msp_wu, msp_vu = msp_eigh(Tensor(np.array(A).astype(np.complex128)))
    assert np.allclose(sym_Al @ msp_vl.asnumpy() - msp_vl.asnumpy() @ np.diag(msp_wl.asnumpy()),
                       np.zeros((n, n)), rtol, atol)
    assert np.allclose(sym_Au @ msp_vu.asnumpy() - msp_vu.asnumpy() @ np.diag(msp_wu.asnumpy()),
                       np.zeros((n, n)), rtol, atol)
    # test for real scalar complex128 no vector
    msp_eigh = EighNet(False, True)
    msp_wl0 = msp_eigh(Tensor(np.array(A).astype(np.complex128)))
    msp_eigh = EighNet(False, False)
    msp_wu0 = msp_eigh(Tensor(np.array(A).astype(np.complex128)))
    assert np.allclose(msp_wl.asnumpy() - msp_wl0.asnumpy(), np.zeros((n, n)), rtol, atol)
    assert np.allclose(msp_wu.asnumpy() - msp_wu0.asnumpy(), np.zeros((n, n)), rtol, atol)
