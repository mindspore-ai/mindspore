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

from mindspore import Tensor
from mindspore import context
from mindspore import dtype as mstype
from mindspore.nn import Cell


class NetWorkFancyIndexBoolean(Cell):
    def __init__(self, index):
        super(NetWorkFancyIndexBoolean, self).__init__()
        self.index = index

    def construct(self, tensor):
        return tensor[self.index]


class NetWorkFancyIndexInterger(Cell):
    def __init__(self, index):
        super(NetWorkFancyIndexInterger, self).__init__()
        self.index = index

    def construct(self, tensor):
        return tensor[self.index]


class NetWorkFancyIndexIntergerBooleanMixed(Cell):
    def __init__(self, index):
        super(NetWorkFancyIndexIntergerBooleanMixed, self).__init__()
        self.index = index

    def construct(self, tensor):
        return tensor[self.index]


def test_tensor_fancy_index_integer_list():
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True)
    index = [0, 2, 1]
    net = NetWorkFancyIndexBoolean(index)
    input_np = np.arange(60).reshape(3, 4, 5)
    input_me = Tensor(input_np, dtype=mstype.float32)
    output_me = net(input_me).asnumpy()
    output_np = input_np[index]
    assert np.allclose(output_np, output_me, 0, 0)


def test_tensor_fancy_boolean_list():
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True)
    index = [True, True, False]
    net = NetWorkFancyIndexInterger(index)
    input_np = np.arange(60).reshape(3, 4, 5)
    input_me = Tensor(input_np, dtype=mstype.float32)
    output_me = net(input_me).asnumpy()
    output_np = input_np[index]
    assert np.allclose(output_np, output_me, 0, 0)


def test_tensor_fancy_integer_boolean_list():
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True)
    index = [1, 2, True, False]
    net = NetWorkFancyIndexIntergerBooleanMixed(index)
    input_np = np.arange(60).reshape(3, 4, 5)
    input_me = Tensor(input_np, dtype=mstype.float32)
    output_me = net(input_me).asnumpy()
    output_np = input_np[index]
    assert np.allclose(output_np, output_me, 0, 0)
