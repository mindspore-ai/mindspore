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
import os
import numpy as np
from tests.st.compiler.control.cases_register import case_register

import mindspore.nn as nn
from mindspore import context, jit
from mindspore.common.tensor import Tensor
from mindspore.train.serialization import export, load


class SingleWhileNet(nn.Cell):
    def construct(self, x, y):
        x += 1
        while x < y:
            x += 1
        y += 2 * x
        return y


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_single_while():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    network = SingleWhileNet()

    x = Tensor(np.array([1]).astype(np.float32))
    y = Tensor(np.array([2]).astype(np.float32))
    origin_out = network(x, y)

    file_name = "while_net"
    export(network, x, y, file_name=file_name, file_format='MINDIR')
    mindir_name = file_name + ".mindir"
    assert os.path.exists(mindir_name)

    graph = load(mindir_name)
    loaded_net = nn.GraphCell(graph)
    outputs_after_load = loaded_net(x, y)
    assert origin_out == outputs_after_load


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_jit_function_while():
    """
    Features: Control flow.
    Description: Test while in @jit decorated function.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    network = SingleWhileNet()

    x = Tensor(np.array([1]).astype(np.float32))
    y = Tensor(np.array([2]).astype(np.float32))
    origin_out = network(x, y)

    file_name = "while_net"
    export(network, x, y, file_name=file_name, file_format='MINDIR')
    mindir_name = file_name + ".mindir"
    assert os.path.exists(mindir_name)

    graph = load(mindir_name)
    loaded_net = nn.GraphCell(graph)
    context.set_context(mode=context.PYNATIVE_MODE)

    @jit
    def run_graph(x, y):
        outputs = loaded_net(x, y)
        return outputs

    outputs_after_load = run_graph(x, y)
    assert origin_out == outputs_after_load


class SingleWhileInlineNet(nn.Cell):
    def construct(self, x, y):
        x += 1
        while x < y:
            x += 1
        y += x
        return y


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_single_while_inline_export():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    network = SingleWhileInlineNet()

    x = Tensor(np.array([1]).astype(np.float32))
    y = Tensor(np.array([2]).astype(np.float32))

    file_name = "while_inline_net"
    export(network, x, y, file_name=file_name, file_format='MINDIR')
    mindir_name = file_name + ".mindir"
    assert os.path.exists(mindir_name)


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_single_while_inline_load():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    network = SingleWhileInlineNet()

    x = Tensor(np.array([1]).astype(np.float32))
    y = Tensor(np.array([2]).astype(np.float32))

    file_name = "while_inline_net"
    export(network, x, y, file_name=file_name, file_format='MINDIR')
    mindir_name = file_name + ".mindir"
    assert os.path.exists(mindir_name)
    load(mindir_name)


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_single_while_inline():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    network = SingleWhileInlineNet()

    x = Tensor(np.array([1]).astype(np.float32))
    y = Tensor(np.array([2]).astype(np.float32))
    origin_out = network(x, y)

    file_name = "while_inline_net"
    export(network, x, y, file_name=file_name, file_format='MINDIR')
    mindir_name = file_name + ".mindir"
    assert os.path.exists(mindir_name)

    graph = load(mindir_name)
    loaded_net = nn.GraphCell(graph)
    outputs_after_load = loaded_net(x, y)
    assert origin_out == outputs_after_load
