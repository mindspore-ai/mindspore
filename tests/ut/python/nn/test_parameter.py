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
""" test parameter """
import numpy as np
import pytest
from mindspore import Tensor, Parameter, ParameterTuple
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer
from mindspore._checkparam import _check_str_by_regular


def test_parameter_init():
    dat = np.array([[1, 2, 3], [2, 3, 4]])
    tensor = Tensor(dat)
    Parameter(tensor, name="testParameter", requires_grad=True, layerwise_parallel=False)


def test_parameter_tuple_illegal():
    p1 = Parameter(initializer(0, [1], mstype.int32), name="global_step1")
    p2 = Parameter(initializer(0, [1], mstype.int32), name="global_step2")
    plist = [p1,p2]
    plist2 = [p1, "str"]
    ptuple = (p1, p2)
    ptuple_str = ("2", "1")
    pstr = "[2,3]"
    pnum = 3

    ParameterTuple(plist)
    ParameterTuple(ptuple)
    with pytest.raises(TypeError):
        ParameterTuple(p1)
    with pytest.raises(ValueError):
        ParameterTuple(plist2)
    with pytest.raises(ValueError):
        ParameterTuple(ptuple_str)
    with pytest.raises(ValueError):
        ParameterTuple(pstr)
    with pytest.raises(TypeError):
        ParameterTuple(pnum)


def test_parameter_init_illegal():
    import numpy as np
    dat = np.array([[1, 2, 3], [2, 3, 4]])
    tensor = Tensor(dat)
    data_none = None
    data_bool = True
    data_str = "nicai"
    data_int = 3
    data_list = [1, "2", True]
    data_tuple = (1, 2, 3)
    np_arr_int16 = np.ones([1,1], dtype=np.int16)
    np_arr_int32 = np.ones([1,1], dtype=np.int32)
    np_arr_float16 = np.ones([1,1], dtype=np.float16)
    np_arr_float32 = np.ones([1,1], dtype=np.float32)

#    with pytest.raises(ValueError):
#        Parameter(np_arr_int16[0][0], name=data_str)
    Parameter(np_arr_int32[0], name=data_str)
    Parameter(np_arr_float16[0], name=data_str)
    Parameter(np_arr_float32[0], name=data_str)
    Parameter(np_arr_float32, name=data_str)

    Parameter(tensor, name=data_str)
    Parameter(data_int, name=data_str)
    Parameter(dat, name=data_str)
    with pytest.raises(ValueError):
        Parameter(data_none, name=data_str)
    with pytest.raises(ValueError):
        Parameter(data_bool, name=data_str)
    with pytest.raises(ValueError):
        Parameter(data_str, name=data_str)
    Parameter(data_list, name=data_str)
    with pytest.raises(ValueError):
        Parameter(data_tuple, name=data_str)

    Parameter(tensor, name=data_str)
    Parameter(tensor, name=data_none)
    with pytest.raises(ValueError):
        Parameter(tensor, name=dat)
    with pytest.raises(ValueError):
        Parameter(tensor, name=tensor)
    with pytest.raises(ValueError):
        Parameter(tensor, name=data_bool)
    with pytest.raises(ValueError):
        Parameter(tensor, name=data_int)
    with pytest.raises(ValueError):
        Parameter(tensor, name=data_list)
    with pytest.raises(ValueError):
        Parameter(tensor, name=data_tuple)

    Parameter(tensor, name=data_str, requires_grad=data_bool)
    with pytest.raises(TypeError):
        Parameter(tensor, name=data_str, requires_grad=data_none)
    with pytest.raises(TypeError):
        Parameter(tensor, name=data_str, requires_grad=dat)
    with pytest.raises(TypeError):
        Parameter(tensor, name=data_str, requires_grad=tensor)
    with pytest.raises(TypeError):
        Parameter(tensor, name=data_str, requires_grad=data_str)
    with pytest.raises(TypeError):
        Parameter(tensor, name=data_str, requires_grad=data_int)
    with pytest.raises(TypeError):
        Parameter(tensor, name=data_str, requires_grad=data_list)
    with pytest.raises(TypeError):
        Parameter(tensor, name=data_str, requires_grad=data_tuple)

    Parameter(tensor, name=data_str, requires_grad=data_bool,layerwise_parallel=data_bool)
    with pytest.raises(TypeError):
        Parameter(tensor, name=data_str, requires_grad=data_bool,layerwise_parallel=dat)
    with pytest.raises(TypeError):
        Parameter(tensor, name=data_str, requires_grad=data_bool,layerwise_parallel=tensor)
    with pytest.raises(TypeError):
        Parameter(tensor, name=data_str, requires_grad=data_bool,layerwise_parallel=data_none)
    with pytest.raises(TypeError):
        Parameter(tensor, name=data_str, requires_grad=data_bool,layerwise_parallel=data_str)
    with pytest.raises(TypeError):
        Parameter(tensor, name=data_str, requires_grad=data_bool,layerwise_parallel=data_int)
    with pytest.raises(TypeError):
        Parameter(tensor, name=data_str, requires_grad=data_bool,layerwise_parallel=data_list)
    with pytest.raises(TypeError):
        Parameter(tensor, name=data_str, requires_grad=data_bool,layerwise_parallel=data_tuple)


def test_check_str_by_regular():
    str1 = "12_sf.asdf_"
    str2 = "x12_sf.asdf."
    str3 = "_x12_sf.asdf"
    str4 = ".12_sf.asdf"
    str5 = "12_sf.a$sdf."
    str6 = "12+sf.asdf"
    _check_str_by_regular(str1)
    _check_str_by_regular(str2)
    _check_str_by_regular(str3)
    with pytest.raises(ValueError):
        _check_str_by_regular(str4)
    with pytest.raises(ValueError):
        _check_str_by_regular(str5)
    with pytest.raises(ValueError):
        _check_str_by_regular(str6)
