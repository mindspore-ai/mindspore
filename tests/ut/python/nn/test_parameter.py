
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

from mindspore import context, Tensor, Parameter, ParameterTuple
from mindspore._checkparam import _check_str_by_regular
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer


def test_parameter_init():
    dat = np.array([[1, 2, 3], [2, 3, 4]])
    tensor = Tensor(dat)
    Parameter(tensor, name="testParameter", requires_grad=True, layerwise_parallel=False)


def test_parameter_tuple_illegal():
    p1 = Parameter(initializer(0, [1], mstype.int32), name="global_step1")
    p2 = Parameter(initializer(0, [1], mstype.int32), name="global_step2")
    plist = [p1, p2]
    plist2 = [p1, "str"]
    ptuple = (p1, p2)
    ptuple_str = ("2", "1")
    pstr = "[2,3]"
    pnum = 3

    ParameterTuple(plist)
    ParameterTuple(ptuple)
    with pytest.raises(TypeError):
        ParameterTuple(p1)
    with pytest.raises(TypeError):
        ParameterTuple(plist2)
    with pytest.raises(TypeError):
        ParameterTuple(ptuple_str)
    with pytest.raises(TypeError):
        ParameterTuple(pstr)
    with pytest.raises(TypeError):
        ParameterTuple(pnum)


def test_parameter_init_illegal():
    dat = np.array([[1, 2, 3], [2, 3, 4]])
    tensor = Tensor(dat)
    data_none = None
    data_bool = True
    data_str = "nicai"
    data_int = 3
    data_list = [1, "2", True]
    data_tuple = (1, 2, 3)

    # test data
    Parameter(tensor, name=data_str)
    Parameter(data_int, name=data_str)
    Parameter(dat, name=data_str)
    with pytest.raises(ValueError):
        Parameter(data_bool, name=data_str)

    # test name
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

    Parameter(tensor, name=data_str, requires_grad=data_bool, layerwise_parallel=data_bool)
    with pytest.raises(TypeError):
        Parameter(tensor, name=data_str, requires_grad=data_bool, layerwise_parallel=dat)
    with pytest.raises(TypeError):
        Parameter(tensor, name=data_str, requires_grad=data_bool, layerwise_parallel=tensor)
    with pytest.raises(TypeError):
        Parameter(tensor, name=data_str, requires_grad=data_bool, layerwise_parallel=data_none)
    with pytest.raises(TypeError):
        Parameter(tensor, name=data_str, requires_grad=data_bool, layerwise_parallel=data_str)
    with pytest.raises(TypeError):
        Parameter(tensor, name=data_str, requires_grad=data_bool, layerwise_parallel=data_int)
    with pytest.raises(TypeError):
        Parameter(tensor, name=data_str, requires_grad=data_bool, layerwise_parallel=data_list)
    with pytest.raises(TypeError):
        Parameter(tensor, name=data_str, requires_grad=data_bool, layerwise_parallel=data_tuple)


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

def test_parameter_compute():
    para_1 = Parameter(initializer('ones', [1, 2, 3], mstype.int32), 'test1')
    para_2 = Parameter(initializer('ones', [1, 2, 3], mstype.int32), 'test2')

    t3 = Tensor(np.ones((1, 2, 3)))

    out = para_1 + para_2
    assert np.array_equal(out.asnumpy(), np.ones((1, 2, 3)) * 2)

    out = para_1 * para_2
    assert np.array_equal(out.asnumpy(), np.ones((1, 2, 3)))

    out = para_1 + t3
    assert np.array_equal(out.asnumpy(), np.ones((1, 2, 3)) * 2)

    out = para_1 * t3
    assert np.array_equal(out.asnumpy(), np.ones((1, 2, 3)))

    assert isinstance(para_1, Tensor)


def test_scalar_parameter_update():
    fp = Parameter(0.5, 'fp')
    fp.default_input = 0.8
    assert np.array_equal(fp.default_input.asnumpy(), np.array(0.8, np.float32))
    fp.default_input = 1
    assert np.array_equal(fp.default_input.asnumpy(), np.array(1.0, np.float32))
    int_ = Parameter(1, 'fp')
    int_.default_input = 2
    assert np.array_equal(int_.default_input.asnumpy(), np.array(2, np.int32))
    with pytest.raises(TypeError):
        int_.default_input = 1.2


def test_parameter_lazy_init():
    # support lazy init in SEMI_AUTO_PARALLEL mode
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8)
    # Call init_data() without set default_input.
    para = Parameter(initializer('ones', [1, 2, 3], mstype.float32), 'test1')
    assert not isinstance(para.default_input, Tensor)
    para = para.init_data()
    assert isinstance(para.default_input, Tensor)
    assert np.array_equal(para.default_input.asnumpy(), np.ones((1, 2, 3)))

    # Call init_data() after default_input is set.
    para = Parameter(initializer('ones', [1, 2, 3], mstype.float32), 'test2')
    assert not isinstance(para.default_input, Tensor)
    # expect type error when not init
    with pytest.raises(TypeError):
        para.default_input = Tensor(np.zeros((1, 2, 3)))
    # init then assign
    para = para.init_data()
    # check the type
    with pytest.raises(TypeError):
        para.default_input = Tensor(np.zeros((1, 2, 3)))
    # check the shape
    with pytest.raises(ValueError):
        para.default_input = Tensor(np.zeros((1, 2)))
    # expect change ok
    para.default_input = Tensor(np.zeros((1, 2, 3)).astype(np.float32))
    assert np.array_equal(para.default_input.asnumpy(), np.zeros((1, 2, 3)))
    para.default_input = initializer('ones', [1, 2, 3], mstype.float32)
    assert isinstance(para.default_input, Tensor)
    # same object and has inited
    assert np.array_equal(para.default_input.asnumpy(), np.ones((1, 2, 3)))
    # expect no effect.
    para.init_data()
    assert np.array_equal(para.default_input.asnumpy(), np.ones((1, 2, 3)))
    para.set_parameter_data(Tensor(np.zeros((1, 2)).astype(np.float32)), slice_shape=True)
    assert np.array_equal(para.default_input.asnumpy(), np.zeros((1, 2)))
    para.set_parameter_data(initializer('ones', [1, 2], mstype.float32), slice_shape=True)
    assert np.array_equal(para.default_input.asnumpy(), np.ones((1, 2)))
    context.reset_auto_parallel_context()
