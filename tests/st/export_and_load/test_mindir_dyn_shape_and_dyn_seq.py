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
"""Export and load mindir in dynamic length of sequence and dynamic shape."""
import os
import pytest

import numpy as np

import mindspore.nn as nn
import mindspore as ms
from mindspore.common import mutable
from mindspore.common.tensor import Tensor
from mindspore.train.serialization import export, load
from mindspore import context

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_dynamic_shape_tuple():
    """
    Feature: export dynamic shape to MindIR file
    Description: Test export API to export network into MindIR
    Expectation: run successfully
    """
    class TestCell(nn.Cell):
        def construct(self, x):
            return x.shape + (1,)

    test_cell = TestCell()
    file_name = "test"
    export(test_cell, Tensor(shape=[None, 2, 3], dtype=ms.float32), file_name=file_name, file_format="MINDIR")
    verify_name = file_name + ".mindir"
    assert os.path.exists(verify_name)

    x = Tensor(input_np_x)

    file_name = "net"
    export(test_cell, x, file_name=file_name, file_format='MINDIR')
    verify_name = file_name + ".mindir"
    assert os.path.exists(verify_name)

    graph = load(verify_name)
    net_mindir = nn.GraphCell(graph)
    result_mindir = net_mindir(x)

    out_net = test_cell(x)
    assert out_net == result_mindir
    os.remove(verify_name)


input_np_x = np.random.rand(2, 3, 3).astype(np.float32)


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def construct(self, x):
        x = x[0] + x[1]
        return self.relu(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_mutable_tuple():
    """
    Feature: export mutable tuple size to MindIR file
    Description: Test export API to export network into MindIR
    Expectation: run successfully
    """
    x = [Tensor(input_np_x), Tensor(input_np_x)]

    net = Net()
    file_name = "net"
    export(net, mutable(x), file_name=file_name, file_format='MINDIR')
    verify_name = file_name + ".mindir"
    assert os.path.exists(verify_name)

    graph = load(verify_name)
    net_mindir = nn.GraphCell(graph)
    result_mindir = net_mindir(mutable(x))
    out_net = net(x)
    assert np.allclose(result_mindir.asnumpy(), out_net.asnumpy(), 0.0001, 0.0001)
    os.remove(verify_name)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_mutable_dynamic_tuple():
    """
    Feature: export dynamic tuple size to MindIR file
    Description: Test export API to export network into MindIR
    Expectation: run successfully
    """
    x = [Tensor(input_np_x), Tensor(input_np_x)]

    y = [Tensor(input_np_x), Tensor(input_np_x), Tensor(input_np_x), Tensor(input_np_x)]
    net = Net()
    file_name = "net"
    export(net, mutable(x, dynamic_len=True), file_name=file_name, file_format='MINDIR')
    verify_name = file_name + ".mindir"
    assert os.path.exists(verify_name)

    graph = load(verify_name)
    net_mindir = nn.GraphCell(graph)
    result_mindir = net_mindir(mutable(x))

    out_net = net(x)
    assert np.allclose(result_mindir.asnumpy(), out_net.asnumpy(), 0.0001, 0.0001)

    out_net = net(y)
    result_mindir = net_mindir(mutable(y))
    assert np.allclose(result_mindir.asnumpy(), out_net.asnumpy(), 0.0001, 0.0001)

    os.remove(verify_name)
