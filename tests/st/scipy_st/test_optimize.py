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
"""st for optimize."""

import pytest
import numpy as onp
import scipy as osp

import mindspore.numpy as mnp
import mindspore.scipy as msp
from mindspore import context
from mindspore.common import Tensor

from .utils import match_array

context.set_context(mode=context.GRAPH_MODE)


def rosenbrock(np):
    def func(x):
        return np.sum(100. * np.diff(x) ** 2 + (1. - x[:-1]) ** 2)

    return func


def himmelblau(np):
    def func(p):
        x, y = p
        return (x ** 2 + y - 11.) ** 2 + (x + y ** 2 - 7.) ** 2

    return func


def matyas(np):
    def func(p):
        x, y = p
        return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y

    return func


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64])
@pytest.mark.parametrize('func_x0', [(rosenbrock, onp.zeros(2)),
                                     (himmelblau, onp.zeros(2)),
                                     (himmelblau, onp.array([92, 0.001])),
                                     (matyas, onp.ones(2))])
def test_bfgs(dtype, func_x0):
    """
    Feature: ALL TO ALL
    Description: test cases for bfgs in PYNATIVE mode
    Expectation: the result match scipy
    """
    func, x0 = func_x0
    x0 = x0.astype(dtype)
    x0_tensor = Tensor(x0)
    ms_res = msp.optimize.minimize(func(mnp), x0_tensor, method='BFGS',
                                   options=dict(maxiter=None, gtol=1e-6)).x
    scipy_res = osp.optimize.minimize(func(onp), x0, method='BFGS').x
    match_array(ms_res.asnumpy(), scipy_res, error=5)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64])
def test_bfgs_fixes4594(dtype):
    """
    Feature: ALL TO ALL
    Description: test cases for bfgs in PYNATIVE mode
    Expectation: the result match scipy
    """
    n = 2
    A = Tensor(onp.eye(n, dtype=dtype)) * 1e4

    def func(x):
        return mnp.mean((mnp.dot(A, x)) ** 2)

    results = msp.optimize.minimize(func, Tensor(onp.ones(n, dtype=dtype)), method='BFGS',
                                    options=dict(maxiter=None, gtol=1e-6)).x
    onp.testing.assert_allclose(results.asnumpy(), onp.zeros(n, dtype=dtype), rtol=1e-6, atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64])
@pytest.mark.parametrize('func_x0', [(rosenbrock, onp.zeros(2)), (rosenbrock, onp.zeros(300))])
def test_bfgs_graph(dtype, func_x0):
    """
    Feature: ALL TO ALL
    Description: test cases for bfgs in GRAPH mode
    Expectation: the result match scipy
    """
    context.set_context(mode=context.GRAPH_MODE)

    func, x0 = func_x0
    x0 = x0.astype(dtype)
    x0_tensor = Tensor(x0)
    ms_res = msp.optimize.minimize(func(mnp), x0_tensor, method='BFGS',
                                   options=dict(maxiter=None, gtol=1e-6)).x
    scipy_res = osp.optimize.minimize(func(onp), x0, method='BFGS').x
    match_array(ms_res.asnumpy(), scipy_res, error=5)
