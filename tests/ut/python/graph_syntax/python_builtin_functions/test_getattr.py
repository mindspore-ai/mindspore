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
"""test graph getattr"""

import os
import pytest
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor, jit, jit_class, context

context.set_context(mode=context.GRAPH_MODE)


def test_getattr_tensor_2():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support tensor input.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = Tensor([-1, -2, -3])
        abs_func = getattr(x, "abs")
        return abs_func()

    out = foo()
    assert np.all(out.asnumpy() == np.array([1, 2, 3]))


def test_getattr_tensor_with_concate_string_2():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support tensor input and concate string.
    Expectation: No exception.
    """

    @jit
    def foo():
        attr_str = "a" + "bs"
        abs_func = getattr(Tensor([-1, -2, -3]), attr_str)
        return abs_func()

    out = foo()
    assert np.all(out.asnumpy() == np.array([1, 2, 3]))


def test_getattr_tensor_with_wrong_attr():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support tensor input.
    Expectation: AttributeError.
    """

    @jit
    def foo(x):
        abs_func = getattr(x, "abs2")
        return abs_func()

    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '0'
    with pytest.raises(AttributeError) as err:
        foo(Tensor([-1, -2, -3]))  # Not throw error any more, should move to ST.
    assert "object has no attribute" in str(err.value)
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '1'


def test_getattr_tensor_with_default():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support tensor input.
    Expectation: No exception.
    """

    @jit
    def foo():
        abs_func = getattr(Tensor([-1, -2, -3]), "abs", Tensor([-1, -2, -3]))
        return abs_func()

    out = foo()
    assert np.all(out.asnumpy() == np.array([1, 2, 3]))


def test_getattr_tensor_with_default_2():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support tensor input.
    Expectation: No exception.
    """

    @jit
    def foo():
        abs_func = getattr(Tensor([-1, -2, -3]), "abs2", Tensor([-1, -2, -3]))
        return abs_func

    out = foo()
    assert np.all(out.asnumpy() == np.array([-1, -2, -3]))


def test_getattr_list():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support list input.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = [1, 2, 3, 4]
        abs_func = getattr(x, "__len__")
        return abs_func()

    out = foo()
    assert out == 4


def test_getattr_list_2():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support list input.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        abs_func = getattr(x, "__len__")
        return abs_func()

    out = foo([1, 2, 3, 4])
    assert out == 4


def test_getattr_list_with_concate_input():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support list input.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = [1, 2, 3, 4]
        attr_str = "__" + "len" + "__"
        abs_func = getattr(x, attr_str)
        return abs_func()

    out = foo()
    assert out == 4


def test_getattr_with_concate_input_2():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support list input.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        attr_str = "__" + "len" + "__"
        abs_func = getattr(x, attr_str)
        return abs_func()

    out = foo([1, 2, 3, 4])
    assert out == 4


def test_getattr_list_with_default():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support list input with default.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = [1, 2, 3, 4]
        abs_func = getattr(x, "__len__", Tensor([-1]))
        return abs_func()

    out = foo()
    assert out == 4


def test_getattr_list_with_default_2():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support list input with default.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = [1, 2, 3, 4]
        abs_func = getattr(x, "__len2__", Tensor([-1]))
        return abs_func

    out = foo()
    assert out == -1


def test_getattr_list_with_wrong_attr():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support list input.
    Expectation: AttributeError.
    """

    @jit
    def foo(x):
        abs_func = getattr(x, "abs2")
        return abs_func()

    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '0'
    with pytest.raises(AttributeError) as err:
        foo([1, 2, 3, 4])  # Not throw error any more, should move to ST.
    assert "object has no attribute" in str(err.value)
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '1'


def test_getattr_tuple():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support tuple input.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = (1, 2, 3, 4)
        abs_func = getattr(x, "__len__")
        return abs_func()

    out = foo()
    assert out == 4


def test_getattr_tuple_2():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support tuple input.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        abs_func = getattr(x, "__len__")
        return abs_func()

    out = foo((1, 2, 3, 4))
    assert out == 4


def test_getattr_tuple_with_concate_input():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support tuple input.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = (1, 2, 3, 4)
        attr_str = "__" + "len" + "__"
        abs_func = getattr(x, attr_str)
        return abs_func()

    out = foo()
    assert out == 4


def test_tuple_getattr_with_concate_input_2():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support tuple input.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        attr_str = "__" + "len" + "__"
        abs_func = getattr(x, attr_str)
        return abs_func()

    out = foo((1, 2, 3, 4))
    assert out == 4


def test_getattr_tuple_with_default():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support tuple input with default.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = (1, 2, 3, 4)
        abs_func = getattr(x, "__len__", Tensor([-1]))
        return abs_func()

    out = foo()
    assert out == 4


def test_getattr_tuple_with_default_2():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support tuple input with default.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = (1, 2, 3, 4)
        abs_func = getattr(x, "__len2__", Tensor([-1]))
        return abs_func

    out = foo()
    assert out == -1


def test_getattr_tuple_with_wrong_attr():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support tensor input.
    Expectation: AttributeError.
    """

    @jit
    def foo(x):
        abs_func = getattr(x, "shape")
        return abs_func()

    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '0'
    with pytest.raises(AttributeError) as err:
        foo((1, 2, 3, 4))  # Not throw error any more, should move to ST.
    assert "object has no attribute" in str(err.value)
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '1'


def test_getattr_dict():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support dict input.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = {"1": 1, "2": 2}
        abs_func = getattr(x, "__len__")
        return abs_func()

    out = foo()
    assert out == 2


def test_getattr_dict_2():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support dict input.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        abs_func = getattr(x, "__len__")
        return abs_func()

    out = foo({"1": 1, "2": 2})
    assert out == 2


def test_getattr_dict_with_concate_input():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support dict input.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = {"1": 1, "2": 2}
        attr_str = "__" + "len" + "__"
        abs_func = getattr(x, attr_str)
        return abs_func()

    out = foo()
    assert out == 2


def test_getattr_dict_with_concate_input_2():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support dict input.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        attr_str = "__" + "len" + "__"
        abs_func = getattr(x, attr_str)
        return abs_func()

    out = foo({"1": 1, "2": 2})
    assert out == 2


def test_getattr_dict_with_wrong_attr():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support tensor input.
    Expectation: AttributeError.
    """

    @jit
    def foo(x):
        abs_func = getattr(x, "abs2")
        return abs_func()

    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '0'
    with pytest.raises(AttributeError) as err:
        foo({"1": 1, "2": 2})  # Not throw error any more, should move to ST.
    assert "object has no attribute" in str(err.value)
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '1'


def test_getattr_dict_with_default():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support dict input with default.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = {"1": 1, "2": 2}
        abs_func = getattr(x, "__len__", Tensor([-1]))
        return abs_func()

    out = foo()
    assert out == 2


def test_getattr_dict_with_default_2():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support dict input with default.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = {"1": 1, "2": 2}
        abs_func = getattr(x, "__len2__", Tensor([-1]))
        return abs_func

    out = foo()
    assert out == -1


@jit_class
class MSClass1:
    def __init__(self):
        self.num0 = Tensor(0)
        self.num1 = Tensor(1)
        self.num2 = Tensor(2)
        self.num3 = Tensor(3)
        self.none = None


def test_getattr_ms_class():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support ms_class input.
    Expectation: No exception.
    """
    ms_obj = MSClass1()

    @jit
    def foo():
        return getattr(ms_obj, "num1")

    out = foo()
    assert out == 1


def test_getattr_ms_class_with_concate_attr():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support ms_class input.
    Expectation: No exception.
    """
    ms_obj = MSClass1()

    @jit
    def foo():
        ret = 0
        nums = ["0", "1", "2", "3"]
        for i in range(4):
            attr_str = "num" + nums[i]
            ret = ret + getattr(ms_obj, attr_str)
        return ret

    out = foo()
    assert out == 6


def test_getattr_ms_class_with_concate_attr_and_default():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support ms_class input.
    Expectation: No exception.
    """
    ms_obj = MSClass1()

    @jit
    def foo():
        ret = 0
        nums = ["0", "1", "2", "3", "4"]
        for i in range(5):
            attr_str = "num" + nums[i]
            ret = ret + getattr(ms_obj, attr_str, Tensor([4]))
        return ret

    out = foo()
    assert out == 10


def test_getattr_ms_class_with_wrong_attr():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support list input.
    Expectation: AttributeError.
    """
    ms_obj = MSClass1()

    @jit
    def foo():
        abs_func = getattr(ms_obj, "abs2")
        return abs_func()

    with pytest.raises(AttributeError) as err:
        foo()
    assert "object has no attribute" in str(err.value)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.a0 = Tensor([0])
        self.a1 = Tensor([1])
        self.a2 = Tensor([2])
        self.a3 = Tensor([3])
        self.none = None

    def construct(self):
        return self.a0


def test_getattr_cell_obj():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support cell object input.
    Expectation: No exception.
    """
    cell_obj = Net()

    @jit
    def foo():
        return getattr(cell_obj, "a0")

    out = foo()
    assert out == 0


def test_getattr_cell_obj_concate_input():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support cell object input.
    Expectation: No exception.
    """
    cell_obj = Net()

    @jit
    def foo():
        a = 0
        attrs = ["0", "1", "2", "3"]
        for attr in attrs:
            a = a + getattr(cell_obj, "a" + attr)
        return a

    out = foo()
    assert out == 6


def test_getattr_cell_obj_concate_input_and_default_value():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support cell object input.
    Expectation: No exception.
    """
    cell_obj = Net()

    @jit
    def foo():
        a = 0
        attrs = ["0", "1", "2", "3", "4"]
        for attr in attrs:
            a = a + getattr(cell_obj, "a" + attr, Tensor([4]))
        return a

    out = foo()
    assert out == 10


def test_getattr_cell_obj_with_wrong_attr():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support tensor input.
    Expectation: AttributeError.
    """

    cell_obj = Net()

    @jit
    def foo():
        abs_func = getattr(cell_obj, "foo")
        return abs_func()

    with pytest.raises(AttributeError) as err:
        foo()
    assert "object has no attribute" in str(err.value)


def test_getattr_numpy_array():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support numpy array input.
    Expectation: TypeError
    """

    @jit
    def foo():
        x = np.array([1, 2, 3, 4])
        return getattr(x, "shape")[0]

    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '0'
    with pytest.raises(TypeError) as err:
        foo()  # Not throw error any more, should move to ST.
    assert "Do not support to get attribute" in str(err.value)
    os.environ['MS_DEV_ENABLE_FALLBACK_RUNTIME'] = '1'


def test_getattr_numpy_array_2():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support numpy array input.
    Expectation: TypeError
    """

    @jit
    def foo():
        x = 1
        return getattr(x, "shape", np.array([0, 1, 2, 3, 4]))

    with pytest.raises(TypeError) as err:
        foo()
    assert "For 'getattr', the third input 'default' can not" in str(err.value)


def test_getattr_for_fg_object():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support function graph object.
    Expectation: No Exception
    """
    @jit_class
    class User:
        def __init__(self):
            self.value = 10

        @jit
        def func(self, t):
            return 2 * t

    class UserNet(nn.Cell):
        def __init__(self):
            super(UserNet, self).__init__()
            self.inner_net = User()

        def construct(self, x):
            return self.inner_net.func(x), getattr(self.inner_net, "tmp", 1)

    context.set_context(mode=context.PYNATIVE_MODE)
    net = UserNet()
    x = Tensor([1, 2, 3])
    net(x)

    context.set_context(mode=context.GRAPH_MODE)
    net(x)
