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
"""st for scipy.optimize."""

import os
import pytest
import numpy as onp
import scipy as osp
from scipy.optimize.linesearch import line_search_wolfe2 as osp_line_search

import mindspore.numpy as mnp
import mindspore.scipy as msp
from mindspore import context
from mindspore.common import Tensor
from mindspore.scipy.optimize.line_search import line_search as msp_line_search
from tests.st.scipy_st.utils import match_array


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
@pytest.mark.parametrize('dtype', [onp.float32])
@pytest.mark.parametrize('func_x0', [(rosenbrock, onp.zeros(2))])
def test_bfgs1(dtype, func_x0):
    """
    Feature: ALL TO ALL
    Description: test cases for bfgs in PYNATIVE mode
    Expectation: the result match scipy
    """
    func, x0 = func_x0
    x0 = x0.astype(dtype)
    x0_tensor = Tensor(x0)
    ms_res = msp.optimize.minimize(func(mnp), x0_tensor, method='BFGS',
                                   options=dict(maxiter=None, gtol=1e-6))
    scipy_res = osp.optimize.minimize(func(onp), x0, method='BFGS')
    match_array(ms_res.x.asnumpy(), scipy_res.x, error=5, err_msg=str(ms_res))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [onp.float32])
@pytest.mark.parametrize('func_x0', [(himmelblau, onp.zeros(2))])
def test_bfgs2(dtype, func_x0):
    """
    Feature: ALL TO ALL
    Description: test cases for bfgs in PYNATIVE mode
    Expectation: the result match scipy
    """
    func, x0 = func_x0
    x0 = x0.astype(dtype)
    x0_tensor = Tensor(x0)
    ms_res = msp.optimize.minimize(func(mnp), x0_tensor, method='BFGS',
                                   options=dict(maxiter=None, gtol=1e-6))
    scipy_res = osp.optimize.minimize(func(onp), x0, method='BFGS')
    match_array(ms_res.x.asnumpy(), scipy_res.x, error=5, err_msg=str(ms_res))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [onp.float32])
@pytest.mark.parametrize('func_x0', [(matyas, onp.ones(2))])
def test_bfgs3(dtype, func_x0):
    """
    Feature: ALL TO ALL
    Description: test cases for bfgs in PYNATIVE mode
    Expectation: the result match scipy
    """
    func, x0 = func_x0
    x0 = x0.astype(dtype)
    x0_tensor = Tensor(x0)
    ms_res = msp.optimize.minimize(func(mnp), x0_tensor, method='BFGS',
                                   options=dict(maxiter=None, gtol=1e-6))
    scipy_res = osp.optimize.minimize(func(onp), x0, method='BFGS')
    match_array(ms_res.x.asnumpy(), scipy_res.x, error=5, err_msg=str(ms_res))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [onp.float64])
@pytest.mark.parametrize('func_x0', [(rosenbrock, onp.zeros(2))])
def test_bfgs4(dtype, func_x0):
    """
    Feature: ALL TO ALL
    Description: test cases for bfgs in PYNATIVE mode
    Expectation: the result match scipy
    """
    func, x0 = func_x0
    x0 = x0.astype(dtype)
    x0_tensor = Tensor(x0)
    ms_res = msp.optimize.minimize(func(mnp), x0_tensor, method='BFGS',
                                   options=dict(maxiter=None, gtol=1e-6))
    scipy_res = osp.optimize.minimize(func(onp), x0, method='BFGS')
    match_array(ms_res.x.asnumpy(), scipy_res.x, error=5, err_msg=str(ms_res))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [onp.float64])
@pytest.mark.parametrize('func_x0', [(himmelblau, onp.zeros(2))])
def test_bfgs5(dtype, func_x0):
    """
    Feature: ALL TO ALL
    Description: test cases for bfgs in PYNATIVE mode
    Expectation: the result match scipy
    """
    func, x0 = func_x0
    x0 = x0.astype(dtype)
    x0_tensor = Tensor(x0)
    ms_res = msp.optimize.minimize(func(mnp), x0_tensor, method='BFGS',
                                   options=dict(maxiter=None, gtol=1e-6))
    scipy_res = osp.optimize.minimize(func(onp), x0, method='BFGS')
    match_array(ms_res.x.asnumpy(), scipy_res.x, error=5, err_msg=str(ms_res))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [onp.float64])
@pytest.mark.parametrize('func_x0', [(matyas, onp.ones(2))])
def test_bfgs6(dtype, func_x0):
    """
    Feature: ALL TO ALL
    Description: test cases for bfgs in PYNATIVE mode
    Expectation: the result match scipy
    """
    func, x0 = func_x0
    x0 = x0.astype(dtype)
    x0_tensor = Tensor(x0)
    ms_res = msp.optimize.minimize(func(mnp), x0_tensor, method='BFGS',
                                   options=dict(maxiter=None, gtol=1e-6))
    scipy_res = osp.optimize.minimize(func(onp), x0, method='BFGS')
    match_array(ms_res.x.asnumpy(), scipy_res.x, error=5, err_msg=str(ms_res))


@pytest.mark.level1
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


@pytest.mark.level1
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
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '0'
    context.set_context(mode=context.GRAPH_MODE)
    func, x0 = func_x0
    x0 = x0.astype(dtype)
    x0_tensor = Tensor(x0)
    ms_res = msp.optimize.minimize(func(mnp), x0_tensor, method='BFGS',
                                   options=dict(maxiter=None, gtol=1e-6))
    scipy_res = osp.optimize.minimize(func(onp), x0, method='BFGS')
    match_array(ms_res.x.asnumpy(), scipy_res.x, error=5, err_msg=str(ms_res))
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '1'


def _scalar_func_1(np):
    def f(x):
        return -x - x ** 3 + x ** 4

    def fprime(x):
        return -1 - 3 * x ** 2 + 4 * x ** 3

    return f, fprime


def _scalar_func_2(np):
    def f(x):
        return np.exp(-4 * x) + x ** 2

    def fprime(x):
        return -4 * np.exp(-4 * x) + 2 * x

    return f, fprime


def _scalar_func_3(np):
    def f(x):
        return -np.sin(10 * x)

    def fprime(x):
        return -10 * np.cos(10 * x)

    return f, fprime


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('maxiter, func, x, p', [(10, _scalar_func_1, 0., 1.),
                                                 (10, _scalar_func_2, 0., 1.),
                                                 (10, _scalar_func_3, 0., 1.)])
def test_scalar_search(maxiter, func, x, p):
    """
    Feature: ALL TO ALL
    Description: test cases for 1-d function
    Expectation: the result match scipy
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    osp_f, osp_fp = func(onp)
    osp_x, osp_p = onp.array(x), onp.array(p)
    osp_res = osp_line_search(osp_f, osp_fp, osp_x, osp_p, maxiter=maxiter)

    msp_f, _ = func(mnp)
    msp_x, msp_p = mnp.array(x), mnp.array(p)
    msp_res = msp_line_search(msp_f, msp_x, msp_p, maxiter=maxiter)

    match_array(msp_res.a_k, osp_res[0], error=5)
    match_array(msp_res.f_k, osp_res[3], error=5)


def _line_func_1(np, *args):
    def f(x):
        return np.dot(x, x)

    def fprime(x):
        return 2 * x

    return f, fprime


def _line_func_2(np, *args):
    def f(x):
        A = args[0]
        return np.dot(x, np.dot(A, x)) + 1

    def fprime(x):
        A = args[0]
        return np.dot(A + A.T, x)

    return f, fprime


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('maxiter, func, x, p',
                         [(10, _line_func_1, [1.13689136, 0.09772497, 0.58295368, -0.39944903, 0.37005589],
                           [-1.30652685, 1.65813068, -0.11816405, -0.6801782, 0.66638308]),
                          (10, _line_func_1, [-0.52118931, -1.84306955, -0.477974, -0.47965581, 0.6203583],
                           [0.69845715, 0.00377089, 0.93184837, 0.33996498, -0.01568211]),
                          (10, _line_func_2, [0.15634897, 1.23029068, 1.20237985, -0.38732682, -0.30230275],
                           [-1.04855297, -1.42001794, -1.70627019, 1.9507754, -0.50965218]),
                          (10, _line_func_2, [0.42833187, 0.06651722, 0.3024719, -0.63432209, -0.36274117],
                           [-0.67246045, -0.35955316, -0.81314628, -1.7262826, 0.17742614])])
def test_line_search(maxiter, func, x, p):
    """
    Feature: ALL TO ALL
    Description: test cases for n-d function in PYNATIVE mode
    Expectation: the result match scipy
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    A = [[1.76405235, 0.40015721, 0.97873798, 2.2408932, 1.86755799],
         [-0.97727788, 0.95008842, -0.15135721, -0.10321885, 0.4105985],
         [0.14404357, 1.45427351, 0.76103773, 0.12167502, 0.44386323],
         [0.33367433, 1.49407907, -0.20515826, 0.3130677, -0.85409574],
         [-2.55298982, 0.6536186, 0.8644362, -0.74216502, 2.26975462]]

    osp_x, osp_p, osp_A = onp.array(x), onp.array(p), onp.array(A)
    osp_f, osp_fp = func(onp, osp_A)
    osp_res = osp_line_search(osp_f, osp_fp, osp_x, osp_p, maxiter=maxiter)

    msp_x, msp_p, msp_A = mnp.array(x), mnp.array(p), mnp.array(A)
    msp_f, _ = func(mnp, msp_A)
    msp_res = msp_line_search(msp_f, msp_x, msp_p, maxiter=maxiter)

    match_array(msp_res.a_k, osp_res[0], error=5)
    match_array(msp_res.f_k, osp_res[3], error=5)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('maxiter, func, x, p',
                         [(10, _line_func_1, [1.13689136, 0.09772497, 0.58295368, -0.39944903, 0.37005589],
                           [-1.30652685, 1.65813068, -0.11816405, -0.6801782, 0.66638308])])
def test_line_search_graph(maxiter, func, x, p):
    """
    Feature: ALL TO ALL
    Description: test cases for n-d function in GRAPH mode
    Expectation: the result match scipy
    """
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '0'
    context.set_context(mode=context.GRAPH_MODE)
    A = [[1.76405235, 0.40015721, 0.97873798, 2.2408932, 1.86755799],
         [-0.97727788, 0.95008842, -0.15135721, -0.10321885, 0.4105985],
         [0.14404357, 1.45427351, 0.76103773, 0.12167502, 0.44386323],
         [0.33367433, 1.49407907, -0.20515826, 0.3130677, -0.85409574],
         [-2.55298982, 0.6536186, 0.8644362, -0.74216502, 2.26975462]]

    osp_x, osp_p, osp_a = onp.array(x), onp.array(p), onp.array(A)
    osp_f, osp_fp = func(onp, osp_a)
    osp_res = osp_line_search(osp_f, osp_fp, osp_x, osp_p, maxiter=maxiter)

    msp_x, msp_p, msp_A = mnp.array(x), mnp.array(p), mnp.array(A)
    msp_f, _ = func(mnp, msp_A)
    msp_res = msp_line_search(msp_f, msp_x, msp_p, maxiter=maxiter)

    match_array(msp_res.a_k, osp_res[0], error=5)
    match_array(msp_res.f_k, osp_res[3], error=5)
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '1'


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [onp.float32])
@pytest.mark.parametrize('func_x0', [(rosenbrock, onp.zeros(2))])
def test_lbfgs1(dtype, func_x0):
    """
    Feature: ALL TO ALL
    Description: test cases for lbfgs in PYNATIVE mode
    Expectation: the result match bfgs
    """
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '0'
    func, x0 = func_x0
    x0 = x0.astype(dtype)
    x0_tensor = Tensor(x0)
    ms_res = msp.optimize.minimize(func(mnp), x0_tensor, method='LBFGS',
                                   options=dict(maxiter=None, gtol=1e-6))
    ma_res = msp.optimize.minimize(func(mnp), x0_tensor, method='BFGS',
                                   options=dict(maxiter=None, gtol=1e-6))
    match_array(ms_res.x.asnumpy(), ma_res.x, error=5, err_msg=str(ms_res))
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '1'


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [onp.float32])
@pytest.mark.parametrize('func_x0', [(himmelblau, onp.zeros(2))])
def test_lbfgs2(dtype, func_x0):
    """
    Feature: ALL TO ALL
    Description: test cases for lbfgs in PYNATIVE mode
    Expectation: the result match bfgs
    """
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '0'
    func, x0 = func_x0
    x0 = x0.astype(dtype)
    x0_tensor = Tensor(x0)
    ms_res = msp.optimize.minimize(func(mnp), x0_tensor, method='LBFGS',
                                   options=dict(maxiter=None, gtol=1e-6))
    ma_res = msp.optimize.minimize(func(mnp), x0_tensor, method='BFGS',
                                   options=dict(maxiter=None, gtol=1e-6))
    match_array(ms_res.x.asnumpy(), ma_res.x, error=5, err_msg=str(ms_res))
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '1'


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [onp.float32])
@pytest.mark.parametrize('func_x0', [(matyas, onp.ones(2))])
def test_lbfgs3(dtype, func_x0):
    """
    Feature: ALL TO ALL
    Description: test cases for lbfgs in PYNATIVE mode
    Expectation: the result match bfgs
    """
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '0'
    func, x0 = func_x0
    x0 = x0.astype(dtype)
    x0_tensor = Tensor(x0)
    ms_res = msp.optimize.minimize(func(mnp), x0_tensor, method='LBFGS',
                                   options=dict(maxiter=None, gtol=1e-6))
    ma_res = msp.optimize.minimize(func(mnp), x0_tensor, method='BFGS',
                                   options=dict(maxiter=None, gtol=1e-6))
    match_array(ms_res.x.asnumpy(), ma_res.x, error=5, err_msg=str(ms_res))
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '1'


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [onp.float64])
@pytest.mark.parametrize('func_x0', [(rosenbrock, onp.zeros(2))])
def test_lbfgs4(dtype, func_x0):
    """
    Feature: ALL TO ALL
    Description: test cases for lbfgs in PYNATIVE mode
    Expectation: the result match bfgs
    """
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '0'
    func, x0 = func_x0
    x0 = x0.astype(dtype)
    x0_tensor = Tensor(x0)
    ms_res = msp.optimize.minimize(func(mnp), x0_tensor, method='LBFGS',
                                   options=dict(maxiter=None, gtol=1e-6))
    ma_res = msp.optimize.minimize(func(mnp), x0_tensor, method='BFGS',
                                   options=dict(maxiter=None, gtol=1e-6))
    match_array(ms_res.x.asnumpy(), ma_res.x, error=5, err_msg=str(ms_res))
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '1'


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [onp.float64])
@pytest.mark.parametrize('func_x0', [(himmelblau, onp.zeros(2))])
def test_lbfgs5(dtype, func_x0):
    """
    Feature: ALL TO ALL
    Description: test cases for lbfgs in PYNATIVE mode
    Expectation: the result match bfgs
    """
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '0'
    func, x0 = func_x0
    x0 = x0.astype(dtype)
    x0_tensor = Tensor(x0)
    ms_res = msp.optimize.minimize(func(mnp), x0_tensor, method='LBFGS',
                                   options=dict(maxiter=None, gtol=1e-6))
    ma_res = msp.optimize.minimize(func(mnp), x0_tensor, method='BFGS',
                                   options=dict(maxiter=None, gtol=1e-6))
    match_array(ms_res.x.asnumpy(), ma_res.x, error=5, err_msg=str(ms_res))
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '1'


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [onp.float64])
@pytest.mark.parametrize('func_x0', [(matyas, onp.ones(2))])
def test_lbfgs6(dtype, func_x0):
    """
    Feature: ALL TO ALL
    Description: test cases for lbfgs in PYNATIVE mode
    Expectation: the result match bfgs
    """
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '0'
    func, x0 = func_x0
    x0 = x0.astype(dtype)
    x0_tensor = Tensor(x0)
    ms_res = msp.optimize.minimize(func(mnp), x0_tensor, method='LBFGS',
                                   options=dict(maxiter=None, gtol=1e-6))
    ma_res = msp.optimize.minimize(func(mnp), x0_tensor, method='BFGS',
                                   options=dict(maxiter=None, gtol=1e-6))
    match_array(ms_res.x.asnumpy(), ma_res.x, error=5, err_msg=str(ms_res))
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '1'


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64])
def test_lbfgs_fixes4594(dtype):
    """
    Feature: ALL TO ALL
    Description: test cases for lbfgs in PYNATIVE mode
    Expectation: the result match bfgs
    """
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '0'
    n = 2
    a = Tensor(onp.eye(n, dtype=dtype)) * 1e4

    def func(x):
        return mnp.mean((mnp.dot(a, x)) ** 2)

    results = msp.optimize.minimize(func, Tensor(onp.ones(n, dtype=dtype)), method='LBFGS',
                                    options=dict(maxiter=None, gtol=1e-6)).x
    onp.testing.assert_allclose(results.asnumpy(), onp.zeros(n, dtype=dtype), rtol=1e-6, atol=1e-6)
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '1'


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64])
@pytest.mark.parametrize('func_x0', [(rosenbrock, onp.zeros(2)), (rosenbrock, onp.zeros(300))])
def test_lbfgs_graph(dtype, func_x0):
    """
    Feature: ALL TO ALL
    Description: test cases for lbfgs in GRAPH mode
    Expectation: the result match bfgs
    """
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '0'
    context.set_context(mode=context.GRAPH_MODE)
    func, x0 = func_x0
    x0 = x0.astype(dtype)
    x0_tensor = Tensor(x0)
    ms_res = msp.optimize.minimize(func(mnp), x0_tensor, method='LBFGS',
                                   options=dict(history_size=150, maxiter=None, gtol=1e-6))
    ma_res = msp.optimize.minimize(func(mnp), x0_tensor, method='BFGS',
                                   options=dict(maxiter=None, gtol=1e-6))
    match_array(ms_res.x.asnumpy(), ma_res.x, error=5, err_msg=str(ms_res))
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '1'
