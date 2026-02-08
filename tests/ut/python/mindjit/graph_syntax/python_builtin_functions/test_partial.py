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
""" test partial"""
from functools import partial

import numpy as np
from mindspore import nn, Tensor, context

context.set_context(mode=context.GRAPH_MODE)


def test_partial_pos_arg():
    """
    Feature: ALL TO ALL
    Description: test cases for partial_pos_arg
    Expectation: the result match given one
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def show(self, x, y, z):
            return x, y, z

        def construct(self, x, y, z):
            f = partial(self.show, x)
            ret = f(y, z)
            return ret

    class Net2(nn.Cell):
        def __init__(self):
            super(Net2, self).__init__()
            self.show = lambda x, y, z: (x, y, z)

        def construct(self, x, y, z):
            f = partial(self.show, x)
            ret = f(y, z)
            return ret

    x = Tensor(np.arange(3).reshape((3,)).astype(np.float32))
    y = Tensor(np.arange(3 * 4).reshape((3, 4)).astype(np.float32))
    z = Tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5)).astype(np.float32))

    for net in [Net(), Net2()]:
        net(x, y, z)


def test_partial_key_ward_arg():
    """
    Feature: ALL TO ALL
    Description: test cases for partial_key_ward_arg
    Expectation: the result match given one
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def show(self, x, y, z):
            return x, y, z

        def construct(self, x, y, z):
            f = partial(self.show, x=x)
            ret = f(y=y, z=z)
            return ret

    class Net2(nn.Cell):
        def __init__(self):
            super(Net2, self).__init__()
            self.show = lambda x, y, z: (x, y, z)

        def construct(self, x, y, z):
            f = partial(self.show, x=x)
            ret = f(y=y, z=z)
            return ret

    x = Tensor(np.arange(3).reshape((3,)).astype(np.float32))
    y = Tensor(np.arange(3 * 4).reshape((3, 4)).astype(np.float32))
    z = Tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5)).astype(np.float32))

    for net in [Net(), Net2()]:
        net(x, y, z)


def test_partial_key_ward_arg_update():
    """
    Feature: ALL TO ALL
    Description: test cases for partial_key_ward_arg_update
    Expectation: the result match given one
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def show(self, x, y, z):
            return x, y, z

        def construct(self, x, y, z):
            f = partial(self.show, x=x, y=y)
            ret = f(y=y, z=z)
            return ret

    class Net2(nn.Cell):
        def __init__(self):
            super(Net2, self).__init__()
            self.show = lambda x, y, z: (x, y, z)

        def construct(self, x, y, z):
            f = partial(self.show, x=x, y=y)
            ret = f(y=y, z=z)
            return ret

    x = Tensor(np.arange(3).reshape((3,)).astype(np.float32))
    y = Tensor(np.arange(3 * 4).reshape((3, 4)).astype(np.float32))
    z = Tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5)).astype(np.float32))

    for net in [Net(), Net2()]:
        net(x, y, z)


def test_partial_key_ward_arg_and_pos_arg():
    """
    Feature: ALL TO ALL
    Description: test cases for partial_key_ward_arg_and_pos_arg
    Expectation: the result match given one
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def show(self, x, y, z):
            return x, y, z

        def construct(self, x, y, z):
            f = partial(self.show, y=y)
            ret = f(2, z=z)
            return ret

    class Net2(nn.Cell):
        def __init__(self):
            super(Net2, self).__init__()
            self.show = lambda x, y, z: (x, y, z)

        def construct(self, x, y, z):
            f = partial(self.show, y=y)
            ret = f(2, z=z)
            return ret

    x = Tensor(np.arange(3).reshape((3,)).astype(np.float32))
    y = Tensor(np.arange(3 * 4).reshape((3, 4)).astype(np.float32))
    z = Tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5)).astype(np.float32))

    for net in [Net(), Net2()]:
        net(x, y, z)


def test_partial_pos_arg_const():
    """
    Feature: ALL TO ALL
    Description: test cases for partial_pos_arg_const
    Expectation: the result match given one
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def show(self, x, y, z):
            return x, y, z

        def construct(self):
            f = partial(self.show, 1)
            ret = f(2, 3)
            return ret

    class Net2(nn.Cell):
        def __init__(self):
            super(Net2, self).__init__()
            self.show = lambda x, y, z: (x, y, z)

        def construct(self):
            f = partial(self.show, 1)
            ret = f(2, 3)
            return ret

    for net in [Net(), Net2()]:
        assert net() == (1, 2, 3)


def test_partial_key_ward_arg_const():
    """
    Feature: ALL TO ALL
    Description: test cases for partial_key_ward_arg_const
    Expectation: the result match given one
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def show(self, x, y, z):
            return x, y, z

        def construct(self):
            f = partial(self.show, x=1)
            ret = f(y=2, z=3)
            return ret

    class Net2(nn.Cell):
        def __init__(self):
            super(Net2, self).__init__()
            self.show = lambda x, y, z: (x, y, z)

        def construct(self):
            f = partial(self.show, x=1)
            ret = f(y=2, z=3)
            return ret

    for net in [Net(), Net2()]:
        assert net() == (1, 2, 3)


def test_partial_key_ward_arg_update_const():
    """
    Feature: ALL TO ALL
    Description: test cases for partial_key_ward_arg_update_const
    Expectation: the result match given one
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def show(self, x, y, z):
            return x, y, z

        def construct(self):
            f = partial(self.show, x=1, y=2)
            ret = f(y=3, z=4)
            return ret

    class Net2(nn.Cell):
        def __init__(self):
            super(Net2, self).__init__()
            self.show = lambda x, y, z: (x, y, z)

        def construct(self):
            f = partial(self.show, x=1, y=2)
            ret = f(y=3, z=4)
            return ret

    for net in [Net(), Net2()]:
        assert net() == (1, 3, 4)


def test_partial_key_ward_arg_and_pos_arg_const():
    """
    Feature: ALL TO ALL
    Description: test cases for partial_key_ward_arg_and_pos_arg_const
    Expectation: the result match given one
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def show(self, x, y, z):
            return x, y, z

        def construct(self):
            f = partial(self.show, y=2)
            ret = f(1, z=3)
            return ret

    class Net2(nn.Cell):
        def __init__(self):
            super(Net2, self).__init__()
            self.show = lambda x, y, z: (x, y, z)

        def construct(self):
            f = partial(self.show, y=2)
            ret = f(1, z=3)
            return ret

    for net in [Net(), Net2()]:
        assert net() == (1, 2, 3)
