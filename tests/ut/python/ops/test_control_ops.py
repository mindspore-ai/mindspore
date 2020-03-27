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
""" test control ops """
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore import context
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE)


def cond_data_test(x_init, y_init):
    class Net(nn.Cell):
        def __init__(self):
            """"""
            super(Net, self).__init__()
            self.square = P.Square()
            self.add = P.TensorAdd()
            self.value = Tensor(np.full((1), 3, dtype=np.float32))
            self.switch = P.GeSwitch()
            self.merge = P.Merge()
            self.less = P.Less()

        def construct(self, x, y):
            cond = self.less(x, y)
            st1, sf1 = self.switch(x, cond)
            st2, sf2 = self.switch(y, cond)
            add_ret = self.add(st1, st2)
            st3, sf3 = self.switch(self.value, cond)
            sq_ret = self.square(sf3)
            ret = self.merge((add_ret, sq_ret))
            return ret[0]

    x = Tensor(x_init, dtype=ms.float32)
    y = Tensor(y_init, dtype=ms.float32)
    net = Net()
    output = net(x, y)
    return output


def test_cond_data_true():
    output = cond_data_test(3, 8)
    print("test_cond_data_true:", output)

def test_cond_data_false():
    output = cond_data_test(8, 3)
    print("test_cond_data_false:", output)

def if_compile_test(x_init, y_init):
    class Net(nn.Cell):
        def __init__(self):
            """"""
            super(Net, self).__init__()
            self.square = P.Square()
            self.add = P.TensorAdd()
            self.value = Tensor(3, dtype=ms.float32)
            self.switch = P.GeSwitch()
            self.merge = P.Merge()
            self.less = P.Less()

        def construct(self, x, y):
            cond = self.less(x, y)
            ret = self.value
            if cond:
                ret = self.add(x, ret)
                ret = self.add(y, ret)
            else:
                ret = self.square(self.value)
            return ret

    x = Tensor(x_init, dtype=ms.float32)
    y = Tensor(y_init, dtype=ms.float32)
    net = Net()
    output = net(x, y)
    return output


def test_if_none():
    class Net(nn.Cell):
        def __init__(self, z: None):
            """"""
            super(Net, self).__init__()
            self.z = z

        def construct(self, x, y):
            if self.z:
                ret = x
            else:
                ret = y
            return ret

    x = Tensor(np.ones([6, 8, 10], np.int32))
    y = Tensor(np.zeros([3, 4, 5], np.int32))
    z = None
    net = Net(z)
    assert net(x, y) == y


def test_if_str_is_not_none_right():
    class Net(nn.Cell):
        def __init__(self, z: str):
            """"""
            super(Net, self).__init__()
            self.z = z

        def construct(self, x, y):
            if self.z == None:
                ret = x
            else:
                ret = y
            return ret

    x = Tensor(np.ones([6, 8, 10], np.int32))
    y = Tensor(np.zeros([3, 4, 5], np.int32))
    z = "ok"
    net = Net(z)
    assert net(x, y) == y


def test_if_str_is_not_none_left():
    class Net(nn.Cell):
        def __init__(self, z: str):
            """"""
            super(Net, self).__init__()
            self.z = z

        def construct(self, x, y):
            if None == self.z:
                ret = x
            else:
                ret = y
            return ret

    x = Tensor(np.ones([6, 8, 10], np.int32))
    y = Tensor(np.zeros([3, 4, 5], np.int32))
    z = "ok"
    net = Net(z)
    assert net(x, y) == y


def test_if_none_equal_none():
    class Net(nn.Cell):
        def __init__(self, z: None):
            """"""
            super(Net, self).__init__()
            self.z = z

        def construct(self, x, y):
            if self.z == None:
                ret = x
            else:
                ret = y
            return ret

    x = Tensor(np.ones([6, 8, 10], np.int32))
    y = Tensor(np.zeros([3, 4, 5], np.int32))
    z = None
    net = Net(z)
    assert net(x, y) == x


def test_if_str_is_null():
    class Net(nn.Cell):
        def __init__(self, z: str):
            """"""
            super(Net, self).__init__()
            self.z = z

        def construct(self, x, y):
            if self.z:
                ret = x
            else:
                ret = y
            return ret

    x = Tensor(np.ones([6, 8, 10], np.int32))
    y = Tensor(np.zeros([3, 4, 5], np.int32))
    z = ""
    net = Net(z)
    assert net(x, y) == y


def test_if_str_is_true():
    class Net(nn.Cell):
        def __init__(self, z: str):
            """"""
            super(Net, self).__init__()
            self.z = z

        def construct(self, x, y):
            if self.z:
                ret = x
            else:
                ret = y
            return ret

    x = Tensor(np.ones([6, 9, 10], np.int32))
    y = Tensor(np.zeros([3, 4, 5], np.int32))
    z = "ok"
    net = Net(z)
    assert net(x, y) == x


def test_if_str_equal():
    class Net(nn.Cell):
        def __init__(self, z: str):
            """"""
            super(Net, self).__init__()
            self.z = z

        def construct(self, x, y):
            if self.z == "ok":
                ret = x
            else:
                ret = y
            return ret

    x = Tensor(np.ones([6, 8, 10], np.int32))
    y = Tensor(np.zeros([3, 4, 5], np.int32))
    z = "ok"
    net = Net(z)
    assert net(x, y) == x


def test_if_tuple_is_null():
    class Net(nn.Cell):
        def __init__(self, z: tuple):
            """"""
            super(Net, self).__init__()
            self.z = z

        def construct(self, x, y):
            if self.z:
                ret = x
            else:
                ret = y
            return ret

    x = Tensor(np.ones([6, 8, 10], np.int32))
    y = Tensor(np.zeros([3, 4, 5], np.int32))
    z = ()
    net = Net(z)
    assert net(x, y) == y


def test_if_tuple_is_not_null():
    class Net(nn.Cell):
        def __init__(self, z: tuple):
            """"""
            super(Net, self).__init__()
            self.z = z

        def construct(self, x, y):
            if self.z:
                ret = x
            else:
                ret = y
            return ret

    x = Tensor(np.ones([6, 8, 10], np.int32))
    y = Tensor(np.zeros([3, 4, 5], np.int32))
    z = (1, 2, 3)
    net = Net(z)
    assert net(x, y) == x


def test_if_dict_is_null():
    class Net(nn.Cell):
        def __init__(self, z: dict):
            """"""
            super(Net, self).__init__()
            self.z = z

        def construct(self, x, y):
            if self.z:
                ret = x
            else:
                ret = y
            return ret

    x = Tensor(np.ones([6, 8, 10], np.int32))
    y = Tensor(np.zeros([3, 4, 5], np.int32))
    z = {}
    net = Net(z)
    assert net(x, y) == y


def test_if_dict_is_not_null():
    class Net(nn.Cell):
        def __init__(self, z: dict):
            """"""
            super(Net, self).__init__()
            self.z = z

        def construct(self, x, y):
            if self.z:
                ret = x
            else:
                ret = y
            return ret

    x = Tensor(np.ones([6, 8, 10], np.int32))
    y = Tensor(np.zeros([3, 4, 5], np.int32))
    z = {"one": 1, "two": 2}
    net = Net(z)
    assert net(x, y) == x


def test_if_else_assign():
    class Net(nn.Cell):
        def __init__(self, m: list):
            """"""
            super(Net, self).__init__()
            self.m = m
            self.n = [4, 5, 6]

        def construct(self, x, y):
            exp_1 = self.m if self.m else self.n
            exp_2 = self.m if exp_1 == self.n else self.n
            if exp_2 == self.m:
                if self.m:
                    ret = x
                else:
                    ret = y
            else:
                if self.m:
                    ret = x
                else:
                    ret = y
            return ret

    x = Tensor(np.ones([6, 8, 10], np.int32))
    y = Tensor(np.zeros([3, 4, 5], np.int32))
    z = [1, 2]
    net = Net(z)
    assert net(x, y) == x


def test_if_compile_true():
    output = if_compile_test(3, 8)
    print("test_if_compile_true:", output)


def test_if_compile_false():
    output = if_compile_test(8, 3)
    print("test_if_compile_false:", output)
