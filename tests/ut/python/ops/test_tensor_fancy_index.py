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
""" test_tensor_slice """
import numpy as np
import pytest

from mindspore import Tensor
from mindspore import context
from mindspore import dtype as mstype
from mindspore.nn import Cell


class NetWorkFancyIndex(Cell):
    def __init__(self, index):
        super(NetWorkFancyIndex, self).__init__()
        self.index = index

    def construct(self, tensor):
        return tensor[self.index]


def test_tensor_fancy_index_integer_list():
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True)
    index = [0, 2, 1]
    net = NetWorkFancyIndex(index)
    input_np = np.arange(60).reshape(3, 4, 5)
    input_me = Tensor(input_np, dtype=mstype.float32)
    net(input_me)


def test_tensor_fancy_index_boolean_list():
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True)
    index = [True, True, False]
    net = NetWorkFancyIndex(index)
    input_np = np.arange(60).reshape(3, 4, 5)
    input_me = Tensor(input_np, dtype=mstype.float32)
    net(input_me)


def test_tensor_fancy_index_integer_boolean_list_graph():
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True)
    index = [1, 2, True, False]
    net = NetWorkFancyIndex(index)
    input_np = np.arange(60).reshape(3, 4, 5)
    input_me = Tensor(input_np, dtype=mstype.float32)
    net(input_me)


def test_tensor_fancy_index_integer_list_mixed():
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True)
    index = (1, [2, 1, 3], slice(1, 3, 1), ..., 4)
    net = NetWorkFancyIndex(index)
    input_np = np.arange(3*4*5*6*7*8).reshape(3, 4, 5, 6, 7, 8)
    input_me = Tensor(input_np, dtype=mstype.float32)
    net(input_me)


def test_tensor_fancy_index_integer_tuple_mixed():
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True)
    index = (1, (2, 1, 3), slice(1, 3, 1), ..., 4)
    net = NetWorkFancyIndex(index)
    input_np = np.arange(3*4*5*6*7*8).reshape(3, 4, 5, 6, 7, 8)
    input_me = Tensor(input_np, dtype=mstype.float32)
    net(input_me)


def test_tensor_fancy_index_integer_list_tuple_mixed():
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True)
    index = (1, [2, 1, 3], (3, 2, 1), slice(1, 3, 1), ..., 4)
    net = NetWorkFancyIndex(index)
    input_np = np.arange(3*4*5*6*7*8).reshape(3, 4, 5, 6, 7, 8)
    input_me = Tensor(input_np, dtype=mstype.float32)
    net(input_me)


def test_tensor_fancy_index_integer_list_tuple_bool_mixed():
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True)
    index = (1, [2, 1, 3], True, (3, 2, 1), slice(1, 3, 1), ..., True, 4)
    net = NetWorkFancyIndex(index)
    input_np = np.arange(3*4*5*6*7*8).reshape(3, 4, 5, 6, 7, 8)
    input_me = Tensor(input_np, dtype=mstype.float32)
    net(input_me)


def test_tensor_fancy_index_integer_list_tuple_bool_mixed_error():
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True)
    index = (1, [2, 1, 3], True, (3, 2, 1), slice(1, 3, 1), ..., False, 4)
    net = NetWorkFancyIndex(index)
    input_np = np.arange(3*4*5*6*7*8).reshape(3, 4, 5, 6, 7, 8)
    input_me = Tensor(input_np, dtype=mstype.float32)
    with pytest.raises(IndexError):
        net(input_me)
