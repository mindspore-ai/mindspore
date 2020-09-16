# Copyright 2020 Huawei Technologies Co., Ltd
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
"""
Test util functions used in distribution classes.
"""
import numpy as np
import pytest

from mindspore.nn.cell import Cell
from mindspore import context
from mindspore import dtype
from mindspore import Tensor
from mindspore.common.parameter import Parameter
from mindspore.nn.probability.distribution._utils.utils import set_param_type, \
    cast_to_tensor, CheckTuple, CheckTensor

def test_set_param_type():
    """
    Test set_param_type function.
    """
    tensor_fp16 = Tensor(0.1, dtype=dtype.float16)
    tensor_fp32 = Tensor(0.1, dtype=dtype.float32)
    tensor_fp64 = Tensor(0.1, dtype=dtype.float64)
    tensor_int32 = Tensor(0.1, dtype=dtype.int32)
    array_fp32 = np.array(1.0).astype(np.float32)
    array_fp64 = np.array(1.0).astype(np.float64)
    array_int32 = np.array(1.0).astype(np.int32)

    dict1 = {'a': tensor_fp32, 'b': 1.0, 'c': tensor_fp32}
    dict2 = {'a': tensor_fp32, 'b': 1.0, 'c': tensor_fp64}
    dict3 = {'a': tensor_int32, 'b': 1.0, 'c': tensor_int32}
    dict4 = {'a': array_fp32, 'b': 1.0, 'c': tensor_fp32}
    dict5 = {'a': array_fp32, 'b': 1.0, 'c': array_fp64}
    dict6 = {'a': array_fp32, 'b': 1.0, 'c': array_int32}
    dict7 = {'a': 1.0}
    dict8 = {'a': 1.0, 'b': 1.0, 'c': 1.0}
    dict9 = {'a': tensor_fp16, 'b': tensor_fp16, 'c': tensor_fp16}
    dict10 = {'a': tensor_fp64, 'b': tensor_fp64, 'c': tensor_fp64}
    dict11 = {'a': array_fp64, 'b': array_fp64, 'c': tensor_fp64}

    ans1 = set_param_type(dict1, dtype.float16)
    assert ans1 == dtype.float32

    with pytest.raises(TypeError):
        set_param_type(dict2, dtype.float32)

    ans3 = set_param_type(dict3, dtype.float16)
    assert ans3 == dtype.float32
    ans4 = set_param_type(dict4, dtype.float16)
    assert ans4 == dtype.float32

    with pytest.raises(TypeError):
        set_param_type(dict5, dtype.float32)
    with pytest.raises(TypeError):
        set_param_type(dict6, dtype.float32)

    ans7 = set_param_type(dict7, dtype.float32)
    assert ans7 == dtype.float32
    ans8 = set_param_type(dict8, dtype.float32)
    assert ans8 == dtype.float32
    ans9 = set_param_type(dict9, dtype.float32)
    assert ans9 == dtype.float16
    ans10 = set_param_type(dict10, dtype.float32)
    assert ans10 == dtype.float32
    ans11 = set_param_type(dict11, dtype.float32)
    assert ans11 == dtype.float32

def test_cast_to_tensor():
    """
    Test cast_to_tensor.
    """
    with pytest.raises(ValueError):
        cast_to_tensor(None, dtype.float32)
    with pytest.raises(TypeError):
        cast_to_tensor(True, dtype.float32)
    with pytest.raises(TypeError):
        cast_to_tensor({'a': 1, 'b': 2}, dtype.float32)
    with pytest.raises(TypeError):
        cast_to_tensor('tensor', dtype.float32)

    ans1 = cast_to_tensor(Parameter(Tensor(0.1, dtype=dtype.float32), 'param'))
    assert isinstance(ans1, Parameter)
    ans2 = cast_to_tensor(np.array(1.0).astype(np.float32))
    assert isinstance(ans2, Tensor)
    ans3 = cast_to_tensor([1.0, 2.0])
    assert isinstance(ans3, Tensor)
    ans4 = cast_to_tensor(Tensor(0.1, dtype=dtype.float32), dtype.float32)
    assert isinstance(ans4, Tensor)
    ans5 = cast_to_tensor(0.1, dtype.float32)
    assert isinstance(ans5, Tensor)
    ans6 = cast_to_tensor(1, dtype.float32)
    assert isinstance(ans6, Tensor)

class Net(Cell):
    """
    Test class: CheckTuple.
    """
    def __init__(self, value):
        super(Net, self).__init__()
        self.checktuple = CheckTuple()
        self.value = value

    def construct(self, value=None):
        if value is None:
            return self.checktuple(self.value, 'input')
        return self.checktuple(value, 'input')

def test_check_tuple():
    """
    Test CheckTuple.
    """
    net1 = Net((1, 2, 3))
    ans1 = net1()
    assert isinstance(ans1, tuple)

    with pytest.raises(TypeError):
        net2 = Net('tuple')
        net2()

    context.set_context(mode=context.GRAPH_MODE)
    net3 = Net((1, 2, 3))
    ans3 = net3()
    assert isinstance(ans3, tuple)

    with pytest.raises(TypeError):
        net4 = Net('tuple')
        net4()

class Net1(Cell):
    """
    Test class: CheckTensor.
    """
    def __init__(self, value):
        super(Net1, self).__init__()
        self.checktensor = CheckTensor()
        self.value = value
        self.context = context.get_context('mode')

    def construct(self, value=None):
        value = self.value if value is None else value
        if self.context == 0:
            self.checktensor(value, 'input')
            return value
        return self.checktensor(value, 'input')

def test_check_tensor():
    """
    Test CheckTensor.
    """
    value = Tensor(0.1, dtype=dtype.float32)
    net1 = Net1(value)
    ans1 = net1()
    assert isinstance(ans1, Tensor)
    ans1 = net1(value)
    assert isinstance(ans1, Tensor)

    with pytest.raises(TypeError):
        net2 = Net1('tuple')
        net2()

    context.set_context(mode=context.GRAPH_MODE)
    net3 = Net1(value)
    ans3 = net3()
    assert isinstance(ans3, Tensor)
    ans3 = net3(value)
    assert isinstance(ans3, Tensor)

    with pytest.raises(TypeError):
        net4 = Net1('tuple')
        net4()
