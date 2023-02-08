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
"""test sparsify"""
import platform
import numpy as np
import pytest
import scipy
import scipy.sparse.linalg
from scipy.linalg import eigvals

from mindspore import Tensor, CSRTensor, context, ops
from mindspore import dtype as mstype
from mindspore.nn import Cell
from mindspore.rewrite import sparsify, ArgType


def to_tensor(obj, tensor_type):
    if tensor_type == "Tensor":
        return Tensor(np.array(obj))
    if tensor_type == "CSRTensor":
        obj = scipy.sparse.csr_matrix(obj)
        return CSRTensor(indptr=Tensor(obj.indptr.astype(np.int32)),
                         indices=Tensor(obj.indices.astype(np.int32)),
                         values=Tensor(obj.data), shape=obj.shape)
    return obj


def create_sym_pos_matrix(shape, dtype):
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(
            'Symmetric positive definite matrix must be a square matrix, but has shape: ', shape)
    n = shape[-1]
    count = 0
    while count < 100:
        x = np.random.random(shape).astype(dtype)
        a = (np.matmul(x, x.T) + np.eye(n)).astype(dtype)
        count += 1
        if np.min(eigvals(a)) > 0:
            return a
    raise ValueError('Symmetric positive definite matrix create failed')


class Norm(Cell):
    def __init__(self):
        # pylint: disable=useless-super-delegation
        super(Norm, self).__init__()

    def construct(self, x):
        return ops.sqrt(ops.reduce_sum(x ** 2))


class Dot(Cell):
    def __init__(self):
        # pylint: disable=useless-super-delegation
        super(Dot, self).__init__()

    def construct(self, a, b):
        b_aligned = ops.reshape(b, (b.shape[0], -1))
        res = ops.matmul(a, b_aligned)
        res = ops.reshape(res, a.shape[:-1] + b.shape[1:])
        return res


class CG(Cell):
    def __init__(self):
        super(CG, self).__init__()
        self.norm = Norm()
        self.dot = Dot()

    def construct(self, a, b, x0, m, maxiter, tol, atol):
        atol = ops.maximum(atol, tol * self.norm(b))

        r = b - self.dot(a, x0)
        z = p = self.dot(m, r)
        rho = self.dot(r, z)
        k = Tensor(0, mstype.int32)
        x = x0
        while k < maxiter and self.norm(r) > atol:
            q = self.dot(a, p)
            alpha = rho / self.dot(p, q)
            x = x + alpha * p
            r = r - alpha * q

            z = self.dot(m, r)
            rho_ = self.dot(r, z)
            beta = rho_ / rho
            p = z + beta * p
            rho = rho_
            k += 1

        cond = self.norm(r) > atol
        return x, ops.select(cond, k, ops.zeros_like(cond).astype(mstype.int32))


def to_np(x):
    if isinstance(x, CSRTensor):
        return scipy.sparse.csr_matrix((x.values.asnumpy(), x.indices.asnumpy(), x.indptr.asnumpy()), shape=x.shape)
    return x.asnumpy()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [context.PYNATIVE_MODE, context.GRAPH_MODE])
@pytest.mark.parametrize("tensor_type_a", ["Tensor", "CSRTensor"])
@pytest.mark.parametrize("tensor_type_m", ["Tensor", "CSRTensor"])
def test_cg(mode, tensor_type_a, tensor_type_m):
    """
    Feature: Sparsify scipy.cg
    Description: test case for sparsify using CG network.
    Expectation: the result matches mindspore.scipy
    """
    if platform.system() != "Linux":
        return
    context.set_context(mode=mode)
    shape = (7, 7)
    dtype = np.float32
    maxiter = 3
    tol = 1e-5
    a = to_tensor(create_sym_pos_matrix(shape, dtype), tensor_type_a)
    np.random.seed(0)
    b = Tensor(np.random.random(shape[:1]).astype(dtype))
    x0 = ops.zeros_like(b)
    m = to_tensor(np.eye(shape[0], dtype=dtype), tensor_type_m)
    sp_res = scipy.sparse.linalg.cg(to_np(a), to_np(b), to_np(x0), M=to_np(m), maxiter=maxiter, atol=tol, tol=tol)

    func = CG()
    arg_types = {}
    if tensor_type_a == "CSRTensor":
        arg_types["a"] = ArgType.CSR
    if tensor_type_m == "CSRTensor":
        arg_types["m"] = ArgType.CSR
    sparse_func = sparsify(func, arg_types)
    sparsify_res = sparse_func(a, b, x0, m, maxiter, tol, tol)

    assert len(sp_res) == len(sparsify_res)
    for expect, actual in zip(sp_res, sparsify_res):
        assert np.allclose(expect, actual.asnumpy(), rtol=1e-3, atol=1e-5)
