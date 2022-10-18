
# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

from mindspore import context, Tensor, Parameter, ParameterTuple, nn
from mindspore._checkparam import Validator
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
    Validator.check_str_by_regular(str1)
    Validator.check_str_by_regular(str2)
    Validator.check_str_by_regular(str3)
    with pytest.raises(ValueError):
        Validator.check_str_by_regular(str4)
    with pytest.raises(ValueError):
        Validator.check_str_by_regular(str5)
    with pytest.raises(ValueError):
        Validator.check_str_by_regular(str6)


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
    # float
    fp = Parameter(0.5, 'fp')
    fp.set_data(0.8)
    assert np.array_equal(fp.data.asnumpy(), np.array(0.8, np.float32))
    fp.set_data(1)
    assert np.array_equal(fp.data.asnumpy(), np.array(1.0, np.float32))
    int_ = Parameter(1, 'fp')
    int_.set_data(2)
    assert np.array_equal(int_.data.asnumpy(), np.array(2, np.int32))
    with pytest.raises(TypeError):
        int_.set_data(1.2)
    # Tensor
    fp32 = Tensor(0.5, mstype.float32)
    int32 = Tensor(2, mstype.int32)
    fp16 = Tensor(0.6, mstype.float16)
    int16 = Tensor(3, mstype.int16)
    bool_ = Tensor(np.array(True, dtype=np.bool_))
    # updata_by_tensor
    fp32_p = Parameter(fp32, 'fp32')
    fp32_p.set_data(0.8)
    fp32_p.set_data(1)
    fp32_p.set_data(int32)
    fp32_p.set_data(fp32)
    fp32_p.set_data(int16)
    fp32_p.set_data(fp16)
    fp32_p.set_data(bool_)

    # updata_by_tensor
    fp16_p = Parameter(fp16, 'fp16')
    with pytest.raises(TypeError):
        fp16_p.set_data(fp32)


def test_parameter_lazy_init():
    # support lazy init in SEMI_AUTO_PARALLEL mode
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8)
    # Call init_data() without set set_data.
    para = Parameter(initializer('ones', [1, 2, 3], mstype.float32), 'test1')
    assert isinstance(para.data, Tensor)
    para = para.init_data()
    assert isinstance(para.data, Tensor)
    assert np.array_equal(para.data.asnumpy(), np.ones((1, 2, 3)))

    para = Parameter(initializer('ones', [1, 2, 3], mstype.complex64), 'test1')
    assert isinstance(para.data, Tensor)
    para = para.init_data()
    assert isinstance(para.data, Tensor)
    assert np.array_equal(para.data.asnumpy(), np.ones((1, 2, 3)))

    # Call init_data() after set_data is set.
    para = Parameter(initializer('ones', [1, 2, 3], mstype.float32), 'test2')
    assert isinstance(para.data, Tensor)
    # expect type error when not init
    with pytest.raises(TypeError):
        para.set_data(Tensor(np.zeros((1, 2, 3))))
    # init then assign
    para = para.init_data()
    # check the type
    with pytest.raises(TypeError):
        para.set_data(Tensor(np.zeros((1, 2, 3))))
    # check the shape
    with pytest.raises(ValueError):
        para.set_data(Tensor(np.zeros((1, 2))))
    # expect change ok
    para.set_data(Tensor(np.zeros((1, 2, 3)).astype(np.float32)))
    assert np.array_equal(para.data.asnumpy(), np.zeros((1, 2, 3)))
    para.set_data(initializer('ones', [1, 2, 3], mstype.float32))
    assert isinstance(para.data, Tensor)
    # same object and has inited
    assert np.array_equal(para.data.asnumpy(), np.ones((1, 2, 3)))
    # expect no effect.
    para.init_data()
    assert np.array_equal(para.data.asnumpy(), np.ones((1, 2, 3)))
    para.set_data(Tensor(np.zeros((1, 2)).astype(np.float32)), slice_shape=True)
    assert np.array_equal(para.data.asnumpy(), np.zeros((1, 2)))
    para.set_data(initializer('ones', [1, 2], mstype.float32), slice_shape=True)
    assert np.array_equal(para.data.asnumpy(), np.ones((1, 2)))
    context.reset_auto_parallel_context()


def test_parameter_as_output():
    """
    Feature: test parameter as output
    Description:
    Expectation: The output has the right data
    """
    initial_input = initializer('One', shape=(2,), dtype=mstype.int32)
    updated_input = Tensor([2, 2], mstype.int32)

    class Net(nn.Cell):
        def __init__(self, initial, updated):
            super().__init__()
            self.initial = initial
            self.updated = updated
            self.p = Parameter(self.initial, name="weight")
            self.new_p = self.p.init_data()
            self.new_p.set_data(self.updated)

        def construct(self):
            return self.new_p

    net = Net(initial_input, updated_input)
    output = net()
    assert np.array_equal(output.asnumpy(), np.array([2, 2], np.int32))


def test_parameter_init_from_tensor():
    """
    Feature: Parameter initialize.
    Description: Parameter initialized from a given tensor, data is shared.
    Expectation: The Parameter and the tensor share same data buffer.
    """
    tensor = Tensor([1], mstype.float32)
    param = Parameter._from_tensor(tensor, name="mypara")  # pylint: disable=W0212
    assert param.name == "mypara"
    assert np.allclose(param.asnumpy(), np.array([1]))
    tensor.asnumpy()[0] = 2
    assert np.allclose(param.asnumpy(), np.array([2]))


def test_parameter_copy():
    """
    Feature: Parameter copy.
    Description: Parameter copy.
    Expectation: The two Parameter's data are the same.
    """
    tensor = Tensor(np.array([[1, 2, 3], [2, 3, 4]]))
    param1 = Parameter(tensor, name="testParameter")
    param2 = param1.copy()
    np.all(param1.data.asnumpy() == param2.data.asnumpy())
