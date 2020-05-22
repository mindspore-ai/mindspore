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
@File   : test_dictionary.py
@Desc   : test_dictionary
"""
import numpy as np

from mindspore import Tensor, context
from mindspore.nn import Cell
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE)


def Xtest_arg_dict():
    class DictNet(Cell):
        """DictNet definition"""

        def __init__(self):
            super(DictNet, self).__init__()
            self.max = P.Maximum()
            self.min = P.Minimum()

        def construct(self, dictionary):
            a = self.max(dictionary["x"], dictionary["y"])
            b = self.min(dictionary["x"], dictionary["y"])
            return a + b

    dictionary = {"x": Tensor(np.ones([3, 2, 3], np.float32)), "y": Tensor(np.ones([1, 2, 3], np.float32))}
    net = DictNet()
    # The outermost network does not support dict parameters,
    # otherwise an exception will be thrown when executing the graph
    net(dictionary)


def test_const_dict():
    class DictNet(Cell):
        """DictNet1 definition"""

        def __init__(self):
            super(DictNet, self).__init__()
            self.max = P.Maximum()
            self.min = P.Minimum()
            self.dictionary = {"x": Tensor(np.ones([3, 2, 3], np.float32)), "y": Tensor(np.ones([1, 2, 3], np.float32))}

        def construct(self):
            a = self.max(self.dictionary["x"], self.dictionary["y"])
            b = self.min(self.dictionary["x"], self.dictionary["y"])
            return a + b

    net = DictNet()
    net()


def test_dict_set_or_get_item():
    class DictNet(Cell):
        """DictNet1 definition"""

        def __init__(self):
            super(DictNet, self).__init__()
            self.dict_ = {"x": 1, "y": 2}
            self.tuple_1 = (1, 2, 3)
            self.tuple_2 = (4, 5, 6)

        def construct(self):
            ret_1 = (88, 99)
            ret_2 = (88, 99)
            self.dict_["x"] = 3
            self.dict_["z"] = 4

            if self.dict_["x"] == 1:
                ret_1 = self.tuple_1
            if self.dict_["z"] == 4:
                ret_2 = self.tuple_2
            ret = ret_1 + ret_2
            return ret

    net = DictNet()
    assert net() == (88, 99, 4, 5, 6)


def test_dict_set_or_get_item_2():
    class DictNet(Cell):
        """DictNet1 definition"""

        def __init__(self):
            super(DictNet, self).__init__()

        def construct(self):
            tuple_1 = (1, 2, 3)
            tuple_2 = (4, 5, 6)
            ret_1 = (88, 99)
            ret_2 = (88, 99)
            dict_ = {"x": 1, "y": 2}
            dict_["x"] = 3
            dict_["z"] = 4

            if dict_["x"] == 1:
                ret_1 = tuple_1
            if dict_["z"] == 4:
                ret_2 = tuple_2
            ret = ret_1 + ret_2
            return ret

    net = DictNet()
    assert net() == (88, 99, 4, 5, 6)


def test_dict_set_or_get_item_3():
    class DictNet(Cell):
        """DictNet1 definition"""

        def __init__(self):
            super(DictNet, self).__init__()
            self.dict_ = {"x": Tensor(np.ones([2, 2, 3], np.float32)), "y": 1}
            self.tensor = Tensor(np.ones([4, 2, 3], np.float32))

        def construct(self):
            self.dict_["y"] = 3
            if self.dict_["y"] == 3:
                self.dict_["x"] = self.tensor
            return self.dict_["x"]

    net = DictNet()
    assert net() == Tensor(np.ones([4, 2, 3], np.float32))


def test_dict_set_item():
    class DictSetNet(Cell):
        def __init__(self):
            super(DictSetNet, self).__init__()
            self.attrs = ("abc", "edf", "ghi", "jkl")

        def construct(self, x):
            my_dict = {"def": x, "abc": x, "edf": x, "ghi": x, "jkl": x}
            for i in range(len(self.attrs)):
                my_dict[self.attrs[i]] = x - i
            return my_dict["jkl"], my_dict["edf"]

    x = Tensor(np.ones([2, 2, 3], np.float32))
    net = DictSetNet()
    out = net(x)
