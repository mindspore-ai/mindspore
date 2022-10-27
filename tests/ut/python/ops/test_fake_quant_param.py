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
""" test fake quant param ops """
import numpy as np
import pytest
import mindspore as ms
import mindspore.context as context
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops.operations import _quant_ops as Q


class FakeQuantParamNet(Cell):
    def __init__(self):
        super().__init__()
        self.ops = Q.FakeQuantParam.linear_quant_param(ms.common.dtype.QuantDtype.INT8, 0.1, 1)

    def construct(self, x):
        return self.ops(x)


@pytest.mark.parametrize('run_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_export(run_mode):
    """
    Feature: test export and load for FakeQuantParam operation
    Description: construct a network with FakeQuantParam, then export and load this network.
    Expectation: export and load successfully.
    """

    context.set_context(mode=run_mode)
    net = FakeQuantParamNet()
    data_in = Tensor(np.ones([1, 1, 32, 32]), ms.float32)
    file_name = "./fake_quant_param.mindir"
    ms.export(net, data_in, file_name=file_name, file_format="MINDIR")
    graph = ms.load(file_name)
    ms.nn.GraphCell(graph)
