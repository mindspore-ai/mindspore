# Copyright 2022-2023 Huawei Technologies Co., Ltd
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

import pytest
import numpy as np
import mindspore as ms
from tests.st.compiler.ms_adapter import Tensor, Parameter
from tests.st.compiler.ms_adapter._register.utils import convert_to_ms_tensor, convert_to_adapter_tensor
from tests.mark_utils import arg_mark


ms.set_context(mode=ms.GRAPH_MODE)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_attr():
    """
    Feature: MSAdapter
    Description: Get the properties of MSAdapter.Tensor
    Expectation: No exception
    """
    @ms.jit
    def func(x):
        return x.attr

    x = Tensor(1)
    assert func(x) == 10


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_method():
    """
    Feature: MSAdapter
    Description: Get the methods of MSAdapter.Tensor
    Expectation: No exception
    """
    @ms.jit
    def func(x):
        return x.method(10)

    x = Tensor(1)
    assert func(x) == 20


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parameter_attr():
    """
    Feature: MSAdapter
    Description: Get the properties of MSAdapter.Parameter
    Expectation: No exception
    """
    @ms.jit
    def func(x):
        return x.attr

    x = Parameter(Tensor(1))
    assert func(x) == 10


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parameter_method():
    """
    Feature: MSAdapter
    Description: Get the methods of MSAdapter.Parameter
    Expectation: No exception
    """
    @ms.jit
    def func(x):
        return x.method(10)

    x = Parameter(Tensor(1))
    assert func(x) == 20


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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

    x = Tensor([1, 2, 3])
    y = ms.Tensor([1, 2, 3])
    out = func(x, y)
    assert type(out[0]) is ms.Tensor
    assert type(out[1]) is Tensor
    assert out[2] == (3, 3, 3, 3)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skip("skip testcase")
def test_convert_to_ms_tensor():
    """
    Feature: MSAdapter
    Description: Test type conversion
    Expectation: No exception
    """
    @ms.jit
    def func(x):
        return convert_to_ms_tensor(x)

    out = func(Tensor([1, 2, 3]))
    assert type(out) is ms.Tensor

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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

    x = Tensor(1)
    out = func(x)
    assert out[0] and not out[1] and out[2]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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

    x = Tensor([1])
    y = Parameter(Tensor([2]), name="val")
    out = func(x, y)
    assert not out[0] and out[1] and out[2]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_create_instance():
    """
    Feature: MSAdapter
    Description: Test isinstance syntax
    Expectation: No exception
    """
    @ms.jit
    def func():
        return Tensor([1])

    out = func()
    assert out == 1

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_adapter_tensor_size():
    """
    Feature: MSAdapter
    Description: Test adapter tensor size
    Expectation: No exception
    """
    @ms.jit
    def func(x):
        return x.size()

    x = Tensor([1, 2, 3, 4])
    out = func(x)
    assert out == (4,)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_adapter_tensor_size_2():
    """
    Feature: MSAdapter
    Description: Test adapter tensor size
    Expectation: No exception
    """
    class Net():
        def __init__(self, type_input):
            super().__init__()
            self.dtype = type_input

        def new_tensor(self, np_input):
            return ms.Tensor(np_input, dtype=self.dtype)

    @ms.jit
    def func():
        net = Net(ms.float32)
        data = np.array([1, 2, 3])
        x = net.new_tensor(data)
        adapter_tensor = convert_to_adapter_tensor(x)
        return adapter_tensor.size()

    out_size = func()
    assert out_size == (3,)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_create_instance_2():
    """
    Feature: MSAdapter
    Description: Test isinstance syntax
    Expectation: No exception
    """
    @ms.jit
    def func(x):
        return Tensor(x+1, dtype=ms.int32)

    out = func(Tensor([1]))
    assert out == 2
