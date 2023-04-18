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
""" test MSAdapter. """

import os
import pytest
import mindspore as ms
from tests.st.ms_adapter import Tensor, Parameter
from tests.st.ms_adapter._register.utils import convert_to_ms_tensor, convert_to_adapter_tensor


ms.set_context(mode=ms.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tensor_attr():
    """
    Feature: MSAdapter
    Description: Get the properties of MSAdapter.Tensor
    Expectation: No exception
    """
    @ms.jit
    def func(x):
        return x.attr

    os.environ['MS_DEV_ENABLE_MS_ADAPTER'] = '1'
    x = Tensor(1)
    assert func(x) == 10
    os.environ['MS_DEV_ENABLE_MS_ADAPTER'] = '0'


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tensor_method():
    """
    Feature: MSAdapter
    Description: Get the methods of MSAdapter.Tensor
    Expectation: No exception
    """
    @ms.jit
    def func(x):
        return x.method(10)

    os.environ['MS_DEV_ENABLE_MS_ADAPTER'] = '1'
    x = Tensor(1)
    assert func(x) == 20
    os.environ['MS_DEV_ENABLE_MS_ADAPTER'] = '0'


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_parameter_attr():
    """
    Feature: MSAdapter
    Description: Get the properties of MSAdapter.Parameter
    Expectation: No exception
    """
    @ms.jit
    def func(x):
        return x.attr

    os.environ['MS_DEV_ENABLE_MS_ADAPTER'] = '1'
    x = Parameter(Tensor(1))
    assert func(x) == 10
    os.environ['MS_DEV_ENABLE_MS_ADAPTER'] = '0'


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_parameter_method():
    """
    Feature: MSAdapter
    Description: Get the methods of MSAdapter.Parameter
    Expectation: No exception
    """
    @ms.jit
    def func(x):
        return x.method(10)

    os.environ['MS_DEV_ENABLE_MS_ADAPTER'] = '1'
    x = Parameter(Tensor(1))
    assert func(x) == 20
    os.environ['MS_DEV_ENABLE_MS_ADAPTER'] = '0'


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tensor_convert_type():
    """
    Feature: MSAdapter
    Description: Test type conversion
    Expectation: No exception
    """
    @ms.jit
    def func(x, y):
        a = x.size(0)
        b = y.size
        x = convert_to_ms_tensor(x)
        y = convert_to_adapter_tensor(y)
        c = x.size
        d = y.size(0)
        return x, y, (a, b, c, d)

    os.environ['MS_DEV_ENABLE_MS_ADAPTER'] = '1'
    x = Tensor([1, 2, 3])
    y = ms.Tensor([1, 2, 3])
    out = func(x, y)
    assert type(out[0]) is ms.Tensor
    assert type(out[1]) is Tensor
    assert out[2] == (3, 3, 3, 3)
    os.environ['MS_DEV_ENABLE_MS_ADAPTER'] = '0'


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tensor_isinstance():
    """
    Feature: MSAdapter
    Description: Test isinstance syntax
    Expectation: No exception
    """
    @ms.jit
    def func(x):
        a = isinstance(x, Tensor)
        x = convert_to_ms_tensor(x)
        b = isinstance(x, Tensor)
        x = convert_to_adapter_tensor(x)
        c = isinstance(x, Tensor)
        return a, b, c

    os.environ['MS_DEV_ENABLE_MS_ADAPTER'] = '1'
    x = Tensor(1)
    out = func(x)
    assert out[0] and not out[1] and out[2]
    os.environ['MS_DEV_ENABLE_MS_ADAPTER'] = '0'


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_parameter_isinstance():
    """
    Feature: MSAdapter
    Description: Test isinstance syntax
    Expectation: No exception
    """
    @ms.jit
    def func(x, y):
        a = isinstance(x, Parameter)
        b = isinstance(y, Tensor)
        c = isinstance(y, Parameter)
        return a, b, c

    os.environ['MS_DEV_ENABLE_MS_ADAPTER'] = '1'
    x = Tensor([1])
    y = Parameter(Tensor([2]), name="val")
    out = func(x, y)
    assert not out[0] and out[1] and out[2]
    os.environ['MS_DEV_ENABLE_MS_ADAPTER'] = '0'
