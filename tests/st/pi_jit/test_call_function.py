# Copyright 2023 Huawei Technologies Co., Ltd
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
"""test bytecode CALL_FUNCTION*"""
import pytest
import numpy as onp
from mindspore import numpy as np
from mindspore import Tensor, jit


def to_numpy_array(data):
    if isinstance(data, (int, tuple)):
        return onp.asarray(data)
    if isinstance(data, Tensor):
        return data.asnumpy()
    return data


def match_array(actual, expected, error=0, err_msg=''):
    actual = to_numpy_array(actual)
    expected = to_numpy_array(expected)
    if error > 0:
        onp.testing.assert_almost_equal(
            actual, expected, decimal=error, err_msg=err_msg)
    else:
        onp.testing.assert_equal(actual, expected, err_msg=err_msg)

def func(x, k=1):
    return x + k

@jit(mode="PIJit")
def jit_test(x):
    y = (x,)
    d = {'k': 10}
    return func(x), func(*y), func(x, k=10), func(x, **d), func(*y, k=10), func(*y, **d)

def python_test(x):
    y = (x,)
    d = {'k': 10}
    return func(x), func(*y), func(x, k=10), func(x, **d), func(*y, k=10), func(*y, **d)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('python_func', [python_test])
@pytest.mark.parametrize('jit_func', [jit_test])
@pytest.mark.parametrize('x', Tensor(np.ones((2, 3)).astype(np.float32)))
def test_call_function(python_func, jit_func, x):
    """
    Feature: test bytecode CALL_FUNCTION/CALL_FUNCTION_KW/CALL_FUNCTION_EX.
    Description: PIJit can support bytecode CALL_FUNCTION/CALL_FUNCTION_KW/CALL_FUNCTION_EX.
    Expectation: The result of PIJit is same as python exe.
    """
    res = python_func(x)
    ms_res = jit_func(x)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
