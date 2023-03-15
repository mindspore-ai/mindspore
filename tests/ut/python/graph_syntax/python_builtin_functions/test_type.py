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
""" test graph fallback buildin python function type"""
import os
import numpy as np
from mindspore import jit, context, Tensor

context.set_context(mode=context.GRAPH_MODE)


def test_fallback_type_with_input_int():
    """
    Feature: JIT Fallback
    Description: Test type() in graph mode with int input.
    Expectation: No exception.
    """
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '0'
    @jit
    def foo():
        x = type(1)
        return x
    out = foo()
    assert str(out) == "<class 'int'>"
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '1'


def test_fallback_type_with_input_float():
    """
    Feature: JIT Fallback
    Description: Test type() in graph mode with float input.
    Expectation: No exception.
    """
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '0'
    @jit
    def foo():
        x = type(1.0)
        return x
    out = foo()
    assert str(out) == "<class 'float'>"
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '1'


def test_fallback_type_with_input_list():
    """
    Feature: JIT Fallback
    Description: Test type() in graph mode with list input.
    Expectation: No exception.
    """
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '0'
    @jit
    def foo():
        x = type([1, 2, 3])
        return x
    out = foo()
    assert str(out) == "<class 'list'>"
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '1'


def test_fallback_type_with_input_tuple():
    """
    Feature: JIT Fallback
    Description: Test type() in graph mode with tuple input.
    Expectation: No exception.
    """
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '0'
    @jit
    def foo():
        x = type((1, 2, 3))
        return x
    out = foo()
    assert str(out) == "<class 'tuple'>"
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '1'


def test_fallback_type_with_input_dict():
    """
    Feature: JIT Fallback
    Description: Test type() in graph mode with dict input.
    Expectation: No exception.
    """
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '0'
    @jit
    def foo():
        x = type({'a': 1, 'b': 2})
        return x
    out = foo()
    assert str(out) == "<class 'dict'>"
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '1'


def test_fallback_type_with_input_numpy_array():
    """
    Feature: JIT Fallback
    Description: Test type() in graph mode with numpy array input.
    Expectation: No exception.
    """
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '0'
    @jit
    def foo():
        x = type(np.array([1, 2, 3]))
        return x
    out = foo()
    assert str(out) == "<class 'numpy.ndarray'>"
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '1'


def test_fallback_type_with_input_tensor():
    """
    Feature: JIT Fallback
    Description: Test type() in graph mode with tensor input.
    Expectation: No exception.
    """
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '0'
    @jit
    def foo():
        x = type(Tensor([1, 2, 3]))
        return x
    out = foo()
    assert str(out) == "<class 'mindspore.common.tensor.Tensor'>"
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '1'
