# Copyright 2025 Huawei Technologies Co., Ltd
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
Construct class and common functions for testing aclgraph feature.
"""
import os
import numpy as np
import mindspore as ms
from mindspore import nn
import mindspore.ops as P
import mindspore.runtime as rt
from mindspore import Tensor, context, jit
from mindspore.common import Parameter
from mindspore import dtype as mstype

context.set_context(
    mode=context.GRAPH_MODE,
    jit_config={
        "jit_level": "O0",
        "infer_boost": "on"
    },
    max_call_depth=600000
)

g_block_num = 20
steps = 20
input_len = 10

class Net(nn.Cell):
    """
    Net definition
    """

    def __init__(self):
        super().__init__()
        self.param = Parameter(Tensor(2, ms.float32))
        self.add = P.Add()
        self.mul = P.Mul()

    def construct(self, x):
        x = self.add(x, self.param)
        for _ in range(5):
            x = self.add(x, 0.1)
            x = self.add(x, 0.2)
        x = self.mul(x, 2)
        x = self.add(x, 0.5)
        return x


class Simple_Net(nn.Cell):
    """
    Simple_Net definition
    """

    def __init__(self):
        super().__init__()
        self.param = Parameter(Tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], mstype.float32))
        self.add = P.Add()

    def construct(self, x):
        output = self.add(x, self.param)
        return output

class SimpleWrapperNet(nn.Cell):
    """
    SimpleWrapperNet definition
    """

    def __init__(self):
        super().__init__()
        self.net = Simple_Net()

    @jit
    def construct(self, x):
        output = self.net(x)
        return output


class SeqNet(nn.Cell):
    """
    SeqNet definition
    """

    def __init__(self):
        super().__init__()
        self.net = Net()

    def construct(self, x):
        output = self.net(x)
        return output

class Net1(nn.Cell):
    """
    Net1 definition
    """

    def __init__(self):
        super().__init__()
        self.add = P.Add()
        self.mul = P.Mul()
        self.sub = P.Sub()
        self.add_n = P.AddN()
        self.reshape = P.Reshape()

    def construct(self, x, key_cache_list, value_cache_list):
        """
        define construct for base network
        """
        y = x
        x = self.reshape(x, (1, -1))
        for i in range(g_block_num):
            key = key_cache_list[int(i/2) % input_len]
            x = self.add(x, 1)
            x = self.sub(x, 1.1)
            x = self.reshape(x, (2, -1))
            x = self.add(x, key)
            x = self.add(x, y)
            x = self.mul(x, 0.251)
            x = self.add(x, 1)
            x = self.add(x, key)
            x = self.mul(x, 0.501)
            x = self.sub(x, 1.1)
            x = self.reshape(x, (2, -1))
            x = self.mul(x, 2)
            x = self.add(x, y)
            x = self.sub(x, 1.1)
            x = self.sub(x, key)
            x = self.reshape(x, (4, -1))
            x = self.mul(x, 0.051)
            x = self.reshape(x, (2, -1))
            x = self.add_n(value_cache_list) + y + x
            x = self.add(x, key)
        x = self.reshape(x, (2, -1))
        return x

def expected_output(x):
    return (x + 3.5) * 2 + 0.5

def run_multi_graph_save():
    """
    py script to test multi graph cache with num limit for capture graph
    """
    rt.set_kernel_launch_capture(True)
    new_input1 = Tensor(np.ones((2, 5)).astype(np.float32))
    new_input2 = Tensor((np.ones((2, 6)) * 2).astype(np.float32))
    new_input3 = Tensor((np.ones((2, 7)) * 3).astype(np.float32))
    new_input4 = Tensor((np.ones((2, 8)) * 4).astype(np.float32))
    dyn_input_data = Tensor(shape=[2, None], dtype=mstype.float32)
    base_shape = (2, 3)

    net = SeqNet()
    net.set_inputs(dyn_input_data)
    net.phase = "increment"

    for i in range(1, 20):
        if i in {5, 6, 9, 12, 15}:
            output = net(new_input1)
            output_np = output.asnumpy()
            expected = 9.5
            assert np.allclose(output_np, expected), \
                f"Output {output_np} does not match expected {expected} at step {i}"
        elif i == 7:
            output = net(new_input2)
            output_np = output.asnumpy()
            expected = 11.5
            assert np.allclose(output_np, expected), \
                f"Output {output_np} does not match expected {expected} at step {i}"
        elif i == 8:
            output = net(new_input4)
            output_np = output.asnumpy()
            expected = 15.5
            assert np.allclose(output_np, expected), \
                f"Output {output_np} does not match expected {expected} at step {i}"
        elif i in {10, 11, 13}:
            output = net(new_input3)
            output_np = output.asnumpy()
            expected = 13.5
            assert np.allclose(output_np, expected), \
                f"Output {output_np} does not match expected {expected} at step {i}"
        else:
            input_data1 = Tensor(np.full(base_shape, i).astype(np.float32))
            output = net(input_data1)
            output_np = output.asnumpy()
            expected = expected_output(i)
            assert np.allclose(output_np, expected), \
                f"Output {output_np} does not match expected {expected} at step {i}"
    command = 'unset MS_DEV_RUNTIME_CONF'
    os.system(command)

if __name__ == "__main__":
    run_multi_graph_save()
