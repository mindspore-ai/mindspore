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
import os
import numpy as np
from tests.st.compiler.control.cases_register import case_register

import mindspore.context as context
from mindspore import Tensor, nn
from mindspore.common import dtype as mstype
from mindspore.train.serialization import export, load


class CaseNet(nn.Cell):
    def __init__(self):
        super(CaseNet, self).__init__()
        self.conv = nn.Conv2d(1, 1, 3)
        self.relu = nn.ReLU()
        self.relu1 = nn.ReLU()
        self.softmax = nn.Softmax()
        self.layers1 = (self.relu, self.softmax)
        self.layers2 = (self.conv, self.relu1)

    def construct(self, x, index1, index2):
        x = self.layers1[index1](x)
        x = self.layers2[index2](x)
        return x


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_mindir_switch_layer():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = CaseNet()
    data = Tensor(np.ones((1, 1, 224, 224)), mstype.float32)
    idx = Tensor(0, mstype.int32)
    idx2 = Tensor(1, mstype.int32)

    file_name = "switch_layer_net"
    mindir_name = file_name + ".mindir"
    export(net, data, idx, idx2, file_name=file_name, file_format='MINDIR')
    assert os.path.exists(mindir_name)

    graph = load(mindir_name)
    loaded_net = nn.GraphCell(graph)
    outputs_after_load = loaded_net(data, idx, idx2)
    relu = nn.ReLU()
    true_value = relu(data)
    ret = np.allclose(outputs_after_load.asnumpy(), true_value.asnumpy())
    assert ret


@case_register.skip(reason="depend on export")
@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_mindir_export():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = CaseNet()
    data = Tensor(np.ones((1, 1, 224, 224)), mstype.float32)
    idx = Tensor(0, mstype.int32)
    idx2 = Tensor(1, mstype.int32)

    file_name = "switch_layer_net"
    mindir_name = file_name + ".mindir"
    export(net, data, idx, idx2, file_name=file_name, file_format='MINDIR')
    assert os.path.exists(mindir_name)


@case_register.skip(reason="depend on export")
@case_register.level1
@case_register.target_ascend
def test_mindir_load():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    data = Tensor(np.ones((1, 1, 224, 224)), mstype.float32)
    idx = Tensor(0, mstype.int32)
    idx2 = Tensor(1, mstype.int32)

    file_name = "switch_layer_net"
    mindir_name = file_name + ".mindir"
    graph = load(mindir_name)
    loaded_net = nn.GraphCell(graph)
    outputs_after_load = loaded_net(data, idx, idx2)
    relu = nn.ReLU()
    true_value = relu(data)
    ret = np.allclose(outputs_after_load.asnumpy(), true_value.asnumpy())
    assert ret
