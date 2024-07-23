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

import time
import numpy as np
from tests.mark_utils import arg_mark
import mindspore
from mindspore import context, ops, nn, Tensor, Parameter


class NetNonConcurrent(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = ops.ReLU()
        self.add = ops.Add()

    def construct(self, input_x, input_x1, input_x2, input_x3):
        output = self.relu(input_x)
        for _ in range(50):
            output = self.add(output, 1)
        for _ in range(50):
            output = self.add(output, 1)
        for _ in range(50):
            output = self.add(output, 1)
        for _ in range(50):
            output = self.add(output, 1)
        return output


class NetNonConcurrentWithWhile(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = ops.ReLU()
        self.add = ops.Add()

    def construct(self, input_x, input_loop, input_x1, input_loop1):
        output = self.relu(input_x)
        while input_loop < 4:
            input_loop = input_loop + 1
            for _ in range(50):
                output = self.add(output, 1)
        return output


class NetConcurrent(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = ops.ReLU()
        self.add = ops.Add()

    def construct(self, input_x1, input_x2, input_x3, input_x4):
        output1 = self.relu(input_x1)
        output2 = self.relu(input_x2)
        output3 = self.relu(input_x3)
        output4 = self.relu(input_x4)
        for _ in range(50):
            output1 = self.add(output1, 1)
        for _ in range(50):
            output2 = self.add(output2, 1)
        for _ in range(50):
            output3 = self.add(output3, 1)
        for _ in range(50):
            output4 = self.add(output4, 1)

        output = output1 + output2 + output3 + output4
        return output


class NetConcurrentWithWhile(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = ops.ReLU()
        self.add = ops.Add()

    def construct(self, input_x1, input_loop1, input_x2, input_loop2):
        output1 = self.relu(input_x1)
        while input_loop1 < 2:
            input_loop1 = input_loop1 + 1
            for _ in range(50):
                output1 = self.add(output1, 1)

        output2 = self.relu(input_x2)
        while input_loop2 < 2:
            input_loop2 = input_loop2 + 1
            for _ in range(50):
                output2 = self.add(output2, 1)

        output = output1 + output2
        return output


class SubNet(nn.Cell):
    def __init__(self, inputx):
        super().__init__()
        self.sub = ops.Sub()
        self.inputx = Parameter(inputx, name="weight")

    def construct(self, inputy):
        output = self.sub(self.inputx, inputy)
        return output


def run_multi_actor_fusion(net_name, net, input1, input2, input3, input4, expect_output):
    context.set_context(mode=context.GRAPH_MODE)
    total_time = 0
    total_count = 0
    for i in range(200):
        time1 = time.time()
        output = net(input1, input2, input3, input4).asnumpy()
        time2 = time.time()
        if i > 1:
            total_count += 1
            total_time += (time2 - time1) * 1000
        assert (output == expect_output).all()
    print(net_name + " avg_time:", total_time/total_count)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_ascend', 'platform_gpu'],
          level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_non_concurrent():
    """
    Feature: Multi actor fusion with non concurrent.
    Description: Test the net which is non concurrent, that can trigger the function of multi actor fusion.
    Expectation: The value and shape of output are the expected values.
    """
    input_x = Tensor(np.ones(2), mindspore.float32)
    net = NetNonConcurrent()
    expect = np.array([201, 201])
    run_multi_actor_fusion("non_concurrent", net, input_x, input_x, input_x, input_x, expect)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'],
          level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_non_concurrent_with_while():
    """
    Feature: Multi actor fusion with non concurrent and while.
    Description: Test the net which is non concurrent with while, that can trigger the function of multi actor fusion.
    Expectation: The value and shape of output are the expected values.
    """
    input_x = Tensor(np.ones(2), mindspore.float32)
    input_loop = Tensor([0], mindspore.float32)
    net = NetNonConcurrentWithWhile()
    expect = np.array([201, 201])
    run_multi_actor_fusion("non_concurrent_with_while", net, input_x, input_loop, input_x, input_loop, expect)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_ascend', 'platform_gpu'],
          level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_concurrent():
    """
    Feature: Multi actor fusion with concurrent.
    Description: Test the net which is concurrent, that can trigger the function of multi actor fusion.
    Expectation: The value and shape of output are the expected values.
    """
    input1 = Tensor(np.ones(2), mindspore.float32)
    input2 = Tensor(np.ones(2), mindspore.float32)
    input3 = Tensor(np.ones(2), mindspore.float32)
    input4 = Tensor(np.ones(2), mindspore.float32)
    net = NetConcurrent()
    expect = np.array([204, 204])
    run_multi_actor_fusion("concurrent", net, input1, input2, input3, input4, expect)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'],
          level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_concurrent_with_while():
    """
    Feature: Multi actor fusion with concurrent and while.
    Description: Test the net which is concurrent with while, that can trigger the function of multi actor fusion.
    Expectation: The value and shape of output are the expected values.
    """
    input1 = Tensor(np.ones(2), mindspore.float32)
    input_loop1 = Tensor([0], mindspore.float32)
    input2 = Tensor(np.ones(2), mindspore.float32)
    input_loop2 = Tensor([0], mindspore.float32)
    net = NetConcurrentWithWhile()
    expect = np.array([202, 202])
    run_multi_actor_fusion("concurrent_with_while", net, input1, input_loop1, input2, input_loop2, expect)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parameter_set_data():
    """
    Feature: Runtime performance optimize of data prepare.
    Description: Test the interface set_data of parameter result.
    Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE)
    inputx = np.ones([2, 3]).astype(np.float32)
    inputy = np.zeros([2, 3]).astype(np.float32)
    net = SubNet(Tensor(inputx))
    net(Tensor(inputy))

    net.inputx.set_data(Tensor(inputy))
    output = net(Tensor(inputy)).asnumpy()
    expect = np.zeros([2, 3]).astype(np.float32)
    assert (output == expect).all()
