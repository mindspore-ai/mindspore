# Copyright 2024 Huawei Technologies Co., Ltd
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

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Tensor
from mindspore.graph.api import jit
from mindspore.common import Parameter
import mindspore.context as context
import time
from tests.mark_utils import arg_mark

jit_mode = "PSJit"

class NetJit(nn.Cell):
    def __init__(self):
        super().__init__()
        self.param = Parameter(Tensor(2, ms.float32))
        self.add = P.Add()
        self.mul = P.Mul()
        self.add_n = P.AddN()
        self.len = 10

    @jit(mode=jit_mode)
    def construct(self, x):
        output = []
        for _ in range(10):
            x = self.add(x, x)
            x = self.add_n([x, x])
            x = self.mul(x, x)
            x = self.add_n([x, x])
            output.append(x)
        return output

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.param = Parameter(Tensor(2, ms.float32))
        self.add = P.Add()
        self.len = 10

    @jit(mode=jit_mode)
    def construct(self, x):
        output = []
        for _ in range(self.len):
            x = self.add(x, x)
            output.append(x)
        return output

class BaseSeqNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.param = Parameter(Tensor(2, ms.float32))
        self.net_jit = NetJit()
        self.net = Net()

    def construct(self):
        pass

class SeqNet(BaseSeqNet):
    def __init__(self):
        super().__init__()
        self.add = P.Add()
        self.add_n = P.AddN()

    def construct(self, x):
        outputs = []
        origin_input = x
        out = self.net(x)
        for item in out:
            outputs.append(self.add(item, item))
        out_jit = self.net_jit(origin_input)
        for item in out_jit:
            outputs.append(self.add(item, item))
        for _ in range(10):
            x = self.add_n(outputs)
            out_jit = self.net_jit(x)
            item = self.add_n(out_jit)
            outputs.append(self.add(item, item))
        output = self.add_n(outputs)
        return output


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'],
          level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_graph_pipeline_grad():
    """
    Feature: graph mode support pipeline
    Description: Test multi output case for single op and graph pipeline.
    Expectation:No exception and result is correct
    """
    input_data = Tensor(np.zeros((5, 5)).astype(np.float32))
    context.set_context(mode=context.PYNATIVE_MODE)
    net1 = SeqNet()
    grad = P.GradOperation()
    grad_fn = grad(net1)
    grad_out = grad_fn(input_data)

    start_time = time.time()
    for _ in range(20):
        grad_out = grad_fn(input_data)
    grad_out.asnumpy()
    end_time = time.time()
    expected_grad = 4092
    assert np.allclose(grad_out[0][0].asnumpy(), expected_grad, 0.0001, 0.0001)
    assert np.allclose(grad_out[0][1].asnumpy(), expected_grad, 0.0001, 0.0001)
    assert np.allclose(grad_out[2][3].asnumpy(), expected_grad, 0.0001, 0.0001)
    print("Finally call grad output: ", grad_out)
    cost_time = end_time - start_time
    print("Total time cost: ", cost_time, flush=True)
