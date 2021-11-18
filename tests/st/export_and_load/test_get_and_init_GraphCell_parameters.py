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

"""test get and init GraphCell parameters"""

import numpy as np
import pytest

from mindspore import nn
from mindspore import context
from mindspore import Tensor, Parameter
from mindspore import export, load, save_checkpoint, load_checkpoint


context.set_context(mode=context.GRAPH_MODE)


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


def load_mindir_and_update_params():
    net = Net()
    mindir_name = "net_0.mindir"
    export(net, input_a, input_b, file_name=mindir_name[:-7], file_format='MINDIR')

    load_net = nn.GraphCell(graph=load(mindir_name))
    ret = load_net(input_a, input_b)
    assert np.array_equal(ret.asnumpy(), np_a * np_b * np_param)

    ckpt_name = "net_0.ckpt"
    save_checkpoint(load_net, ckpt_name)
    params_init = load_checkpoint(ckpt_name)
    load_net_with_new_params = nn.GraphCell(graph=load(mindir_name), params_init=params_init)
    return load_net_with_new_params


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_and_init_graph_cell_parameters():
    """
    Description: load mind ir and update parameters.
    Expectation: generate a graph with updated parameters.
    """
    load_net = load_mindir_and_update_params()
    ret = load_net(input_a, input_b)

    assert np.array_equal(ret.asnumpy(), np_a * np_b * (np_param + 1.0))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_init_graph_cell_parameters_with_wrong_type():
    """
    Description: load mind ir and update parameters with wrong type.
    Expectation: raise a ValueError indicating the params type error.
    """
    net = Net()
    mindir_name = "net_1.mindir"
    export(net, input_a, input_b, file_name=mindir_name[:-7], file_format='MINDIR')

    new_params = {"weight": np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32)}
    with pytest.raises(TypeError) as err:
        graph = load(mindir_name)
        load_net = nn.GraphCell(graph, params_init=new_params)
        load_net(input_a, input_b)

    assert "The key of the 'params_init' must be str, and the value must be Tensor or Parameter" in str(err.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_init_graph_cell_parameters_with_wrong_shape():
    """
    Description: load mind ir and update parameters with wrong tensor shape.
    Expectation: raise a ValueError indicating the tensor shape error.
    """
    net = Net()
    mindir_name = "net_2.mindir"
    export(net, input_a, input_b, file_name=mindir_name[:-7], file_format='MINDIR')

    new_params = {"weight": Parameter(np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32))}
    with pytest.raises(ValueError) as err:
        graph = load(mindir_name)
        load_net = nn.GraphCell(graph, params_init=new_params)
        load_net(input_a, input_b)

    assert "Only support update parameter by Tensor with same shape and dtype as it" in str(err.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_init_graph_cell_parameters_with_wrong_dtype():
    """
    Description: load mind ir and update parameters with wrong tensor dtype.
    Expectation: raise a ValueError indicating the tensor dtype error.
    """
    net = Net()
    mindir_name = "net_3.mindir"
    export(net, input_a, input_b, file_name=mindir_name[:-7], file_format='MINDIR')

    new_params = {"weight": Parameter(np.arange(2 * 3).reshape((2, 3)).astype(np.float64))}
    with pytest.raises(ValueError) as err:
        graph = load(mindir_name)
        load_net = nn.GraphCell(graph, params_init=new_params)
        load_net(input_a, input_b)

    assert "Only support update parameter by Tensor with same shape and dtype as it" in str(err.value)
