# Copyright 2021 Huawei Technologies Co., Ltd
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

"""test init GraphCell parameters with illegal data"""

import os

import numpy as np
import pytest

from mindspore import Tensor, Parameter
from mindspore import context
from mindspore import export, load
from mindspore import nn


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.flag = False
        self.weight = Parameter(np_param, requires_grad=True)
        self.dense = nn.Dense(3, 4)

    def construct(self, x, y):
        if self.flag:
            ret = self.dense(x * self.weight)
        else:
            ret = x * y * self.weight
        self.weight += 1.0
        return ret


np_a = np.ones((2, 3), np.float32) + 2
np_b = np.ones((2, 3), np.float32) + 3
np_param = np.arange(2 * 3).reshape((2, 3)).astype(np.float32)
input_a = Tensor(np_a)
input_b = Tensor(np_b)


def remove_generated_file(file_name):
    if os.path.isfile(file_name):
        os.remove(file_name)


def test_init_graph_cell_parameters_with_wrong_type():
    """
    Description: load mind ir and update parameters with wrong type.
    Expectation: raise a ValueError indicating the params_init type error.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    mindir_name = "net_0.mindir"
    export(net, input_a, input_b, file_name=mindir_name[:-7], file_format='MINDIR')

    new_params = np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32)
    with pytest.raises(TypeError) as err:
        graph = load(mindir_name)
        load_net = nn.GraphCell(graph, params_init=new_params)
        load_net(input_a, input_b)

    assert "For 'GraphCell', the argument 'params_init' must be a dict, but got" in str(err.value)
    remove_generated_file(mindir_name)


def test_init_graph_cell_parameters_with_wrong_value_type():
    """
    Description: load mind ir and update parameters with wrong value type.
    Expectation: raise a ValueError indicating the params value type error.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    mindir_name = "net_1.mindir"
    export(net, input_a, input_b, file_name=mindir_name[:-7], file_format='MINDIR')

    new_params = {"weight": np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32)}
    with pytest.raises(TypeError) as err:
        graph = load(mindir_name)
        load_net = nn.GraphCell(graph, params_init=new_params)
        load_net(input_a, input_b)

    assert "For 'GraphCell', the key of the 'params_init' must be str, " \
           "and the value must be Tensor or Parameter" in str(err.value)
    remove_generated_file(mindir_name)


def test_init_graph_cell_parameters_with_wrong_value_shape():
    """
    Description: load mind ir and update parameters with wrong tensor shape.
    Expectation: raise a ValueError indicating the update value shape error.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    mindir_name = "net_2.mindir"
    export(net, input_a, input_b, file_name=mindir_name[:-7], file_format='MINDIR')

    new_params = {"weight": Parameter(np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32))}
    with pytest.raises(ValueError) as err:
        graph = load(mindir_name)
        load_net = nn.GraphCell(graph, params_init=new_params)
        load_net(input_a, input_b)

    assert "Only support update parameter by Tensor or Parameter with same shape and dtype as it" in str(err.value)
    remove_generated_file(mindir_name)


def test_init_graph_cell_parameters_with_wrong_value_dtype():
    """
    Description: load mind ir and update parameters with wrong tensor dtype.
    Expectation: raise a ValueError indicating the update value dtype error.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    mindir_name = "net_3.mindir"
    export(net, input_a, input_b, file_name=mindir_name[:-7], file_format='MINDIR')

    new_params = {"weight": Tensor(np.arange(2 * 3).reshape((2, 3)).astype(np.float64))}
    with pytest.raises(ValueError) as err:
        graph = load(mindir_name)
        load_net = nn.GraphCell(graph, params_init=new_params)
        load_net(input_a, input_b)

    assert "Only support update parameter by Tensor or Parameter with same shape and dtype as it" in str(err.value)
    remove_generated_file(mindir_name)
