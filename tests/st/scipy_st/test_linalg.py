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
"""st for scipy.linalg."""

from typing import Generic
import pytest
import numpy as onp
import scipy as osp

import mindspore.scipy as msp
from mindspore import context, Tensor
import mindspore.numpy as mnp
from .utils import match_array, create_full_rank_matrix, create_sym_pos_matrix

onp.random.seed(0)
context.set_context(mode=context.PYNATIVE_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('args', [(), (1,), (7, -1), (3, 4, 5),
                                  (onp.ones((3, 4), dtype=onp.float32), 5, onp.random.randn(5, 2).astype(onp.float32))])
def test_block_diag(args):
    """
    Feature: ALL TO ALL
    Description: test cases for block_diag
    Expectation: the result match scipy
    """
    tensor_args = tuple([Tensor(arg) for arg in args])
    ms_res = msp.linalg.block_diag(*tensor_args)

    scipy_res = osp.linalg.block_diag(*args)
    match_array(ms_res.asnumpy(), scipy_res)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64])
@pytest.mark.parametrize('shape', [(4, 4), (50, 50), (2, 5, 5)])
def test_inv(dtype, shape):
    """
    Feature: ALL TO ALL
    Description: test cases for inv
    Expectation: the result match numpy
    """
    onp.random.seed(0)
    x = create_full_rank_matrix(shape, dtype)

    ms_res = msp.linalg.inv(Tensor(x))
    scipy_res = onp.linalg.inv(x)
    match_array(ms_res.asnumpy(), scipy_res, error=3)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [4, 5, 6])
@pytest.mark.parametrize('lower', [True, False])
@pytest.mark.parametrize('dtype', [onp.float64])
def test_cholesky(n: int, lower: bool, dtype: Generic):
    """
    Feature: ALL TO ALL
    Description:  test cases for cholesky [N,N]
    Expectation: the result match scipy cholesky
    """
    a = create_sym_pos_matrix((n, n), dtype)
    tensor_a = Tensor(a)
    rtol = 1.e-5
    atol = 1.e-8
    osp_c = osp.linalg.cholesky(a, lower=lower)
    msp_c = msp.linalg.cholesky(tensor_a, lower=lower)
    assert onp.allclose(osp_c, msp_c.asnumpy(), rtol=rtol, atol=atol)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [4, 5, 6])
@pytest.mark.parametrize('lower', [True, False])
@pytest.mark.parametrize('dtype', [onp.float64])
def test_cho_factor(n: int, lower: bool, dtype: Generic):
    """
    Feature: ALL TO ALL
    Description:  test cases for cholesky [N,N]
    Expectation: the result match scipy cholesky
    """
    a = create_sym_pos_matrix((n, n), dtype)
    tensor_a = Tensor(a)
    msp_c, _ = msp.linalg.cho_factor(tensor_a, lower=lower)
    if lower:
        msp_reconstruct_a = mnp.dot(mnp.tril(msp_c), mnp.tril(msp_c).T)
    else:
        msp_reconstruct_a = mnp.dot(mnp.triu(msp_c).T, mnp.triu(msp_c))
    assert onp.allclose(a, msp_reconstruct_a.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [4, 5, 6])
@pytest.mark.parametrize('lower', [True, False])
@pytest.mark.parametrize('dtype', [onp.float64])
def test_cholesky_solver(n: int, lower: bool, dtype):
    """
    Feature: ALL TO ALL
    Description:  test cases for cholesky  solver [N,N]
    Expectation: the result match scipy cholesky_solve
    """
    a = create_sym_pos_matrix((n, n), dtype)
    b = onp.ones((n, 1), dtype=dtype)
    tensor_a = Tensor(a)
    tensor_b = Tensor(b)
    osp_c, lower = osp.linalg.cho_factor(a, lower=lower)
    msp_c, msp_lower = msp.linalg.cho_factor(tensor_a, lower=lower)
    osp_factor = (osp_c, lower)

    ms_cho_factor = (msp_c, msp_lower)
    osp_x = osp.linalg.cho_solve(osp_factor, b)
    msp_x = msp.linalg.cho_solve(ms_cho_factor, tensor_b)
    # pre tensor_a has been inplace.
    tensor_a = Tensor(a)
    assert onp.allclose(onp.dot(a, osp_x), mnp.dot(tensor_a, msp_x).asnumpy())
