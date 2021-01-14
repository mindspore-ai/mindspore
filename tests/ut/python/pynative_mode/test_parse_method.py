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
@File  : test_parse_method.py
@Author:
@Date  : 2019-06-27
@Desc  : test parse the object's method
"""
import logging
from dataclasses import dataclass

import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import context
from mindspore._extends.parse.standard_method import ms_len
from mindspore.common.api import ms_function
from mindspore.common.tensor import Tensor
from mindspore.ops.composite import core
from mindspore.ops.primitive import constexpr
from mindspore.ops import functional as F
from ..ut_filter import non_graph_engine


def setup_module(module):
    context.set_context(mode=context.PYNATIVE_MODE)


log = logging.getLogger("test")
log.setLevel(level=logging.ERROR)


@ms_function
def default_parameter_f(x, y=3):
    """ default_parameter_f """
    z = x + y
    return z


# Test case: test parse fn that use default parameter
def test_parse_defalut_parameter_case1():
    """ Test default parameter function call """
    log.debug("begin test_parse_defalut_parameter_case1")
    ret = default_parameter_f(2)
    log.debug("finished test_parse_defalut_parameter_case1, ret = %r", ret)


def get_val_fn(x):
    """ get_val_fn """
    ret = x + 3
    return ret


# Test case: test bool not
@ms_function
def bool_exp(x, y):
    """ bool_exp """
    return not x > y


def test_bool_exp():
    """ test_bool_exp """
    bool_exp(1, 2)


# Test case: use the variable parameter for @mindspore
@ms_function
def var_parameter_f(x, *args):
    """ var_parameter_f """
    z = x + args[0] + args[1] + args[2]
    return z


def test_var_parameter_case1():
    """ test_var_parameter_case1 """
    log.debug("start test_var_parameter_case1")
    var_parameter_f(1, 2, 3, 4, 5)
    log.debug("end test_var_parameter_case1")


class Net(nn.Cell):
    """ Net definition """

    def __init__(self, value1):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(0)
        self.axis = 0
        self.TC = ClassTest("test_class", 1.2)
        self.value = value1

    @ms_function
    def construct(self, x):
        x = self.get_test_value(x)
        return x

    def get_test_value(self, x):
        ret = x + self.value
        return ret


class ClassTest:
    """ ClassTest definition """

    def __init__(self, name, value1):
        self.name = name
        self.value = value1

    def get_name(self):
        return self.name

    def get_value(self, inc):
        ret = self.value + inc
        return ret

    def __call__(self, *args, **kwargs):
        pass


# Test: call method on parse graph code
@non_graph_engine
def test_call_method_on_construct():
    """ test_call_method_on_construct """
    log.debug("begin test_call_method_on_construct")

    x = Tensor(np.array([[1, 2, 3], [1, 2, 3]]).astype(np.int32))
    y = Tensor(np.array([[2, 3, 4], [1, 1, 2]]).astype(np.int32))
    z = np.array([[3, 5, 7], [2, 3, 5]]).astype(np.int32)

    net = Net(y)
    output = net.construct(x)
    result = output.asnumpy()
    print(result)
    assert np.all(result == z)

    log.debug("finished test_call_method_on_construct")


# Test: call method on parse graph code
class Net1(nn.Cell):
    """ Net1 definition """

    def __init__(self, v1, v2):
        super(Net1, self).__init__()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(0)
        self.axis = 0
        self.TC = ClassTest("test_class", v1)
        self.value = v2

    @ms_function
    def construct(self, x):
        x = x + self.TC.get_value(self.value)
        return x


@non_graph_engine
def test_call_other_object_method():
    """ test_call_other_object_method """
    log.debug("begin test_call_other_object_method")

    x = Tensor(np.array([[1, 2, 3], [1, 2, 3]]).astype(np.int32))
    y = Tensor(np.array([[2, 3, 4], [1, 1, 2]]).astype(np.int32))
    y1 = Tensor(np.array([[5, 4, 5], [1, 1, 2]]).astype(np.int32))
    z = np.array([[8, 9, 12], [3, 4, 7]]).astype(np.int32)

    net = Net1(y, y1)
    with pytest.raises(TypeError):
        output = net.construct(x)
        result = output.asnumpy()
        print(result)
        assert np.all(result == z)

    log.debug("finished test_call_other_object_method")


# Test: call global object method(not self) on parse graph code
value = Tensor(np.array([[3, 4, 5], [1, 1, 2]]).astype(np.int32))
TC = ClassTest("test_class", value)


class Net2(nn.Cell):
    """ Net2 definition """

    def __init__(self, value1):
        super(Net2, self).__init__()
        self.value = value1

    @ms_function
    def construct(self, x):
        x = x + TC.get_value(self.value)
        return x

    @ms_function
    def construct1(self, x):
        x = x + TC.value
        x = x + self.value
        return x


@non_graph_engine
def test_call_no_self_other_object_method():
    """ test_call_no_self_other_object_method """
    log.debug("begin test_call_other_object_method")
    x = Tensor(np.array([[1, 2, 3], [1, 2, 3]]).astype(np.int32))
    y = Tensor(np.array([[2, 3, 4], [1, 1, 2]]).astype(np.int32))
    z = np.array([[6, 9, 12], [3, 4, 7]]).astype(np.int32)

    net = Net2(y)
    with pytest.raises(TypeError):
        output = net.construct(x)
        result = output.asnumpy()
        print(result)
        assert np.all(result == z)

    log.debug("finished test_call_other_object_method")


def test_call_no_self_other_object_attr_value():
    """ test_call_no_self_other_object_attr_value """
    # do not support tensor as init input.
    return


# Test case: use the * to unlock the varargs for @mindspore
def vararg1(x, y):
    """ vararg1 """
    z = x + y
    return z


def varargs_main(fn):
    """ varargs_main """

    @ms_function
    def t1(*args):
        return fn(*args)

    return t1


def test_var_parameter_case3():
    """ test_var_parameter_case3 """
    log.debug("start test_var_parameter_case3")
    ret = varargs_main(vararg1)(1, 2)
    log.debug("ret = %r", ret)
    log.debug("end test_var_parameter_case3")


# Test case: test the flag set
@core(tg=True)
def set_flag(x):
    """ set_flag """
    return x + 1


@ms_function
def set_test_flag_main(x, y):
    """ set_test_flag_main """
    z = set_flag(x)
    z = z + y
    return z


def test_set_flag():
    """ Test default parameter function call """
    log.debug("begin test_set_flag")
    ret = set_test_flag_main(2, 3)
    log.debug("finished test_set_flag, ret = %r", ret)


@dataclass
class Access:
    a: int
    b: int

    def max(self):
        if self.a > self.b:
            return self.a
        return self.b


@ms_function
def invoke_dataclass(x, y):
    """ invoke_dataclass """
    acs = Access(x, y)
    return acs.max()


def test_access():
    """ test_access """
    invoke_dataclass(1, 2)

@dataclass
class Access2:
    a: int
    b: int

    def max(self):
        if self.a > self.b:
            return self.c
        return self.b


@ms_function
def invoke_dataclass2(x, y):
    """ invoke_dataclass """
    acs = Access2(x, y)
    return acs.max()


def test_access_attr_error():
    """ test_access """
    with pytest.raises(AttributeError):
        invoke_dataclass2(2, 1)


def myfunc(x):
    """ myfunc """
    return x * x


@ms_function
def ms_infer_for():
    """ ms_infer_for """
    a = 0.0
    for x in [1.1, 2.3, 3.3]:
        a = a + x
    return a


def test_infer_for():
    """ test_infer_for """
    ms_infer_for()


@ms_function
def ms_infer_for_func(y):
    """ ms_infer_for_func """
    for x in [1.0, 2.0, 3.0]:
        y = myfunc(x) + y
    return y


def test_ms_infer_for_func():
    """ test_ms_infer_for_func """
    ms_infer_for_func(1.0)


@ms_function
def add(x, y):
    """ add """
    return x + y


def test_add():
    """ test_add """
    res = add(1, 2.0)
    return res


@ms_function
def add_list():
    """ add_list """
    a = [1, 2, 3]
    b = a[1] + a[2]
    return b


def test_list():
    """ test_list """
    return add_list()


@ms_function
def compare_list_len():
    """ compare_list_len """
    a = [1, 2, 3]
    return ms_len(a)


def test_list_len():
    """ test_list_len """
    compare_list_len()


@ms_function
def add_tuple():
    """ add_tuple """
    a = (1, 2, 3)
    b = a[1] + a[2]
    return b


def test_tuple():
    """ test_tuple """
    return add_tuple()


def invoke_func(x):
    """ invoke_func """
    return x * x


@ms_function
def tuple_of_node(x, y):
    """ tuple_of_node """
    a = invoke_func(x)
    b = invoke_func(y)
    c = (a, b)
    d = c[1] * x
    return d


def test_tuple_node():
    """ test_tuple_node """
    res = tuple_of_node(1, 2)
    return res


@ms_function
def range_spec(x, y):
    """ range_spec """
    for _ in range(1, 10, 3):
        x = x + 1
    return x + y


def test_range():
    """ test_range """
    res = range_spec(10, 10)
    return res

def test_expr():
    """ test const expr """
    a = (1, 2)
    @constexpr
    def tuple_len(x):
        assert len(x) == 2
    tuple_len(a)


def test_tuple_to_array():
    """ test range tuple to array """
    range_x = range(10)
    res = F.tuple_to_array(range_x)
    print(res)
