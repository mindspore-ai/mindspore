# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""utility functions for mindspore.scipy st tests"""
import platform
from typing import List
from functools import cmp_to_key

import numpy as onp
import scipy.sparse.linalg
from scipy.linalg import eigvals
from mindspore import Tensor, CSRTensor
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore.common import dtype as mstype


def to_tensor(obj, dtype=None, indice_dtype=onp.int32):
    """
    This function is used to initialize Tensor or CSRTensor.
    'obj' can be three type:
        1. tuple or list
            Must be the format: (list, str), and str should be 'Tensor' or 'CSRTensor'.
        2. numpy.ndarray
        3. scipy.sparse.csr_matrix
    """
    if isinstance(obj, (tuple, list)):
        obj, tensor_type = obj
        if tensor_type == "Tensor":
            obj = onp.array(obj)
        elif tensor_type == "CSRTensor":
            obj = scipy.sparse.csr_matrix(obj)

    if dtype is None:
        dtype = obj.dtype

    if isinstance(obj, onp.ndarray):
        obj = Tensor(obj.astype(dtype))
    elif isinstance(obj, scipy.sparse.csr_matrix):
        obj = CSRTensor(indptr=Tensor(obj.indptr.astype(indice_dtype)),
                        indices=Tensor(obj.indices.astype(indice_dtype)),
                        values=Tensor(obj.data.astype(dtype)),
                        shape=obj.shape)

    return obj


def to_ndarray(obj, dtype=None):
    if isinstance(obj, Tensor):
        obj = obj.asnumpy()
    elif isinstance(obj, CSRTensor):
        obj = scipy.sparse.csr_matrix((obj.values.asnumpy(), obj.indices.asnumpy(), obj.indptr.asnumpy()),
                                      shape=obj.shape)
        obj = obj.toarray()

    if dtype is not None:
        obj = obj.astype(dtype)
    return obj


def match_array(actual, expected, error=0, err_msg=''):
    if isinstance(actual, int):
        actual = onp.asarray(actual)

    if isinstance(expected, (int, tuple)):
        expected = onp.asarray(expected)

    if error > 0:
        onp.testing.assert_almost_equal(actual, expected, decimal=error, err_msg=err_msg)
    else:
        onp.testing.assert_equal(actual, expected, err_msg=err_msg)


def match_matrix(actual, expected, error=0, err_msg=''):
    if actual.shape != expected.shape:
        raise ValueError(
            err_msg.join(f" actual shape {actual.shape} is not equal to expected input shape {expected.shape}"))
    sub_abs = mnp.abs(mnp.subtract(actual, expected))
    no_zero_max = sub_abs.max()
    if no_zero_max > Tensor(error, dtype=mstype.float64):
        raise ValueError(
            err_msg.join(f" actual value: {actual} is not equal to expected input value: {expected}"))


def create_full_rank_matrix(shape, dtype):
    if len(shape) < 2 or shape[-1] != shape[-2]:
        raise ValueError(
            'Full rank matrix must be a square matrix, but has shape: ', shape)

    invertible = False
    a = None
    while not invertible:
        a = onp.random.random(shape).astype(dtype)
        try:
            onp.linalg.inv(a)
            invertible = True
        except onp.linalg.LinAlgError:
            pass

    return a


def create_random_rank_matrix(shape, dtype):
    if dtype in [onp.complex64, onp.complex128]:
        random_data = onp.random.uniform(low=-1.0, high=1.0, size=shape).astype(dtype)
        random_data += 1j * onp.random.uniform(low=-1.0, high=1.0, size=shape).astype(dtype)
    elif dtype in [onp.int32, onp.int64]:
        random_data = onp.random.randint(10000, size=shape).astype(dtype)
    else:
        random_data = onp.random.random(shape).astype(dtype)
    return random_data


def create_sym_pos_matrix(shape, dtype):
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(
            'Symmetric positive definite matrix must be a square matrix, but has shape: ', shape)

    n = shape[-1]
    count = 0
    while count < 100:
        x = onp.random.random(shape).astype(dtype)
        a = (onp.matmul(x, x.T) + onp.eye(n)).astype(dtype)
        count += 1
        if onp.min(eigvals(a)) > 0:
            return a
    raise ValueError('Symmetric positive definite matrix create failed')


def gradient_check(x, net, epsilon=1e-3, symmetric=False, enumerate_fn=onp.ndenumerate):
    # Some utils
    def _tensor_to_numpy(arg: List[Tensor]) -> List[onp.ndarray]:
        return [_arg.asnumpy() for _arg in arg]

    def _numpy_to_tensor(arg: List[onp.ndarray]) -> List[Tensor]:
        return [Tensor(_arg) for _arg in arg]

    def _add_value(arg: List[onp.ndarray], outer, inner, value):
        arg[outer][inner] += value
        return arg

    def _flatten(arg: List[onp.ndarray]) -> onp.ndarray:
        arg = [_arg.reshape((-1,)) for _arg in arg]
        return onp.concatenate(arg)

    if isinstance(x, Tensor):
        x = [x]

    # Using automatic differentiation to calculate gradient
    grad_net = ops.GradOperation(get_all=True)(net)
    x_grad = grad_net(*x)
    x_grad = _tensor_to_numpy(x_grad)

    # Using the definition of a derivative to calculate gradient
    x = _tensor_to_numpy(x)
    x_grad_approx = [onp.zeros_like(_x) for _x in x_grad]
    for outer, _x in enumerate(x):
        for inner, _ in enumerate_fn(_x):
            x = _add_value(x, outer, inner, epsilon)
            y_plus = net(*_numpy_to_tensor(x)).asnumpy()

            x = _add_value(x, outer, inner, -2 * epsilon)
            y_minus = net(*_numpy_to_tensor(x)).asnumpy()

            y_grad = (y_plus - y_minus) / (2 * epsilon)
            x = _add_value(x, outer, inner, epsilon)
            x_grad_approx = _add_value(x_grad_approx, outer, inner, y_grad)

    if symmetric:
        x_grad_approx = [0.5 * (_x_grad + _x_grad.conj().T) for _x_grad in x_grad_approx]
    x_grad = _flatten(x_grad)
    x_grad_approx = _flatten(x_grad_approx)
    numerator = onp.linalg.norm(x_grad - x_grad_approx)
    denominator = onp.linalg.norm(x_grad) + onp.linalg.norm(x_grad_approx)
    difference = numerator / denominator
    return difference


def compare_eigen_decomposition(src_res, tgt_res, compute_v, rtol, atol):
    def my_argsort(w):
        """
        Sort eigenvalues, by comparing the real part first, and then the image part
        when the real part is comparatively same (less than rtol).
        """

        def my_cmp(x_id, y_id):
            x = w[x_id]
            y = w[y_id]
            if abs(onp.real(x) - onp.real(y)) < rtol:
                return onp.imag(x) - onp.imag(y)
            return onp.real(x) - onp.real(y)

        w_ind = list(range(len(w)))
        w_ind.sort(key=cmp_to_key(my_cmp))
        return w_ind

    sw, mw = src_res[0], tgt_res[0]
    s_perm = my_argsort(sw)
    m_perm = my_argsort(mw)
    sw = onp.take(sw, s_perm, -1)
    mw = onp.take(mw, m_perm, -1)
    assert onp.allclose(sw, mw, rtol=rtol, atol=atol)

    if compute_v:
        sv, mv = src_res[1], tgt_res[1]
        sv = onp.take(sv, s_perm, -1)
        mv = onp.take(mv, m_perm, -1)

        # Normalize eigenvectors.
        phases = onp.sum(sv.conj() * mv, -2, keepdims=True)
        sv = phases / onp.abs(phases) * sv
        assert onp.allclose(sv, mv, rtol=rtol, atol=atol)


def get_platform():
    return platform.system().lower()
