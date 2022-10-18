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
"""test graph hasattr"""

import numpy as np

import mindspore.nn as nn
from mindspore import Tensor, jit, jit_class, context

context.set_context(mode=context.GRAPH_MODE)


def test_hasattr_tensor():
    """
    Feature: Syntax hasattr.
    Description: Graph syntax hasattr support tensor input.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        return hasattr(x, "abs")

    assert foo(Tensor([-1, -2, -3]))


def test_hasattr_tensor_2():
    """
    Feature: Syntax hasattr.
    Description: Graph syntax hasattr support tensor input.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        return hasattr(x, "abs2")

    assert not foo(Tensor([-1, -2, -3]))


def test_hasattr_list():
    """
    Feature: Syntax hasattr.
    Description: Graph syntax hasattr support list input.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = [1, 2, 3, 4]
        attr = "__" + "len" + "__"
        return hasattr(x, attr)

    assert foo()


def test_hasattr_list_2():
    """
    Feature: Syntax hasattr.
    Description: Graph syntax hasattr support list input.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = [1, 2, 3, 4]
        attr = "__" + "len2" + "__"
        return hasattr(x, attr)

    assert not foo()


def test_hasattr_tuple():
    """
    Feature: Syntax hasattr.
    Description: Graph syntax hasattr support tuple input.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = (1, 2, 3, 4)
        attr = "__" + "len" + "__"
        return hasattr(x, attr)

    assert foo()


def test_hasattr_tuple_2():
    """
    Feature: Syntax hasattr.
    Description: Graph syntax hasattr support tuple input.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = (1, 2, 3, 4)
        attr = "__" + "len2" + "__"
        return hasattr(x, attr)

    assert not foo()


def test_hasattr_dict():
    """
    Feature: Syntax hasattr.
    Description: Graph syntax hasattr support dict input.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = {"1": 1, "2": 2}
        attr = "__" + "len" + "__"
        return hasattr(x, attr)

    assert foo()


def test_hasattr_dict_2():
    """
    Feature: Syntax hasattr.
    Description: Graph syntax hasattr support dict input.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = {"1": 1, "2": 2}
        attr = "__" + "len2" + "__"
        return hasattr(x, attr)

    assert not foo()


@jit_class
class MSClass1:
    def __init__(self):
        self.num0 = Tensor(0)
        self.num1 = Tensor(1)
        self.num2 = Tensor(2)
        self.num3 = Tensor(3)
        self.none = None


def test_hasattr_ms_class():
    """
    Feature: Syntax hasattr.
    Description: Graph syntax hasattr support ms_class input.
    Expectation: No exception.
    """
    ms_obj = MSClass1()

    @jit
    def foo():
        return hasattr(ms_obj, "num1")

    assert foo()


def test_hasattr_ms_class_2():
    """
    Feature: Syntax hasattr.
    Description: Graph syntax hasattr support ms_class input.
    Expectation: No exception.
    """
    ms_obj = MSClass1()

    @jit
    def foo():
        return hasattr(ms_obj, "none")

    assert foo()


def test_hasattr_ms_class_3():
    """
    Feature: Syntax hasattr.
    Description: Graph syntax hasattr support ms_class input.
    Expectation: No exception.
    """
    ms_obj = MSClass1()

    @jit
    def foo():
        return hasattr(ms_obj, "none2")

    assert not foo()


def test_hasattr_ms_class_with_concate_attr():
    """
    Feature: Syntax hasattr.
    Description: Graph syntax hasattr support ms_class input.
    Expectation: No exception.
    """
    ms_obj = MSClass1()

    @jit
    def foo():
        ret = 0
        nums = ["0", "1", "2", "3", "4"]
        for i in range(5):
            attr_str = "num" + nums[i]
            if hasattr(ms_obj, attr_str):
                ret = ret + 1
        return ret

    out = foo()
    assert out == 4


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


def test_hasattr_cell_obj():
    """
    Feature: Syntax hasattr.
    Description: Graph syntax hasattr support cell object input.
    Expectation: No exception.
    """
    cell_obj = Net()

    @jit
    def foo():
        return hasattr(cell_obj, "a0")

    assert foo()


def test_hasattr_cell_obj_2():
    """
    Feature: Syntax hasattr.
    Description: Graph syntax hasattr support cell object input.
    Expectation: No exception.
    """
    cell_obj = Net()

    @jit
    def foo():
        return hasattr(cell_obj, "none")

    assert foo()


def test_hasattr_cell_obj_3():
    """
    Feature: Syntax hasattr.
    Description: Graph syntax hasattr support cell object input.
    Expectation: No exception.
    """
    cell_obj = Net()

    @jit
    def foo():
        return hasattr(cell_obj, "none2")

    assert not foo()


def test_hasattr_cell_obj_concate_input():
    """
    Feature: Syntax hasattr.
    Description: Graph syntax hasattr support cell object input.
    Expectation: No exception.
    """
    cell_obj = Net()

    @jit
    def foo():
        a = 0
        attrs = ["0", "1", "2", "3", "4"]
        for attr in attrs:
            if hasattr(cell_obj, "a" + attr):
                a = a + 1
        return a

    out = foo()
    assert out == 4


def test_hasattr_numpy_array():
    """
    Feature: Syntax hasattr.
    Description: Graph syntax hasattr support numpy array input.
    Expectation: TypeError
    """

    @jit
    def foo():
        x = np.array([1, 2, 3, 4])
        return hasattr(x, "shape")

    assert foo()


def test_hasattr_numpy_array_2():
    """
    Feature: Syntax hasattr.
    Description: Graph syntax hasattr support numpy array input.
    Expectation: TypeError
    """

    @jit
    def foo():
        x = np.array([1, 2, 3, 4])
        return hasattr(x, "shape2")

    assert not foo()
