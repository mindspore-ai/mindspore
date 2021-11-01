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


class Eigh(PrimitiveWithInfer):
    """
    Eigh decomposition
    Ax = lambda * x
    """

    @prim_attr_register
    def __init__(self, compute_eigenvectors):
        super().__init__(name="Eigh")
        self.init_prim_io_names(inputs=['A', 's'], outputs=['output', 'output_v'])
        self.compute_eigenvectors = validator.check_value_type(
            "compute_eigenvectors", compute_eigenvectors, [bool], self.name)

    def __infer__(self, A, s):
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
        return shape


class EighNet(nn.Cell):
    def __init__(self, b):
        super(EighNet, self).__init__()
        self.b = b
        self.eigh = Eigh(b)

    def construct(self, A, s=0):
        r = self.eigh(A, s)
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
def test_eigh_net(n: int, mode):
    """
    Feature: ALL To ALL
    Description: test cases for eigen decomposition test cases for Ax= lambda * x /( A- lambda * E)X=0
    Expectation: the result match to numpy
    """
    context.set_context(mode=mode, device_target="CPU")
    rtol = 1e-4
    atol = 1e-5
    msp_eigh = EighNet(True)
    A = create_sym_pos_matrix(n, n, np.float32)
    tensor_a = Tensor(np.array(A).astype(np.float32))
    msp_w, msp_v = msp_eigh(tensor_a, -1)
    assert np.allclose(A @ msp_v.asnumpy() - msp_v.asnumpy() @ np.diag(msp_w.asnumpy()), np.zeros((n, n)), rtol, atol)

    A = np.random.rand(n, n)
    rtol = 1e-5
    atol = 1e-8
    msp_eigh = EighNet(True)
    msp_w, msp_v = msp_eigh(Tensor(np.array(A).astype(np.float64)), 0)
    msp_wl, msp_vl = msp_eigh(Tensor(np.array(A).astype(np.float64)), -1)
    msp_wu, msp_vu = msp_eigh(Tensor(np.array(A).astype(np.float64)), 1)

    # Compare with scipy
    # sp_w, sp_v = sp.linalg.eig(A.astype(np.float64))
    # sp_wu, sp_vu = sp.linalg.eigh(A.astype(np.float64), lower=False, eigvals_only=False)
    # p_wl, sp_vl = sp.linalg.eigh(np.tril(A).astype(np.float64), lower=True, eigvals_only=False)

    sym_Al = (np.tril((np.tril(A) - np.tril(A).T)) + np.tril(A).T)
    sym_Au = (np.triu((np.triu(A) - np.triu(A).T)) + np.triu(A).T)
    assert np.allclose(sym_Al @ msp_vl.asnumpy() - msp_vl.asnumpy() @ np.diag(msp_wl.asnumpy()), np.zeros((n, n)), rtol,
                       atol)
    assert np.allclose(sym_Au @ msp_vu.asnumpy() - msp_vu.asnumpy() @ np.diag(msp_wu.asnumpy()), np.zeros((n, n)), rtol,
                       atol)
    assert np.allclose(A @ msp_v.asnumpy() - msp_v.asnumpy() @ np.diag(msp_w.asnumpy()), np.zeros((n, n)), rtol, atol)
