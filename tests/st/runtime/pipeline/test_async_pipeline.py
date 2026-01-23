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
# ============================================================================

import time
import numpy as np
import pytest
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Tensor
from mindspore.graph.api import jit
from mindspore import mutable
from mindspore.common import Parameter
from mindspore import dtype as mstype
from tests.mark_utils import arg_mark


g_block_num = 50
inner_block_num = 10
steps = 50


class NetJit(nn.Cell):
    """
    Construct a single-input network structure.
    """
    def __init__(self):
        super().__init__()
        self.param = Parameter(Tensor(2, ms.float32))
        self.add = P.Add()
        self.mul = P.Mul()
        self.add_n = P.AddN()

    @jit(backend="ms_backend")
    def construct(self, x):
        output = []
        for _ in range(g_block_num):
            x = self.add(x, x)
            x = self.add_n([x, x])
            x = self.mul(x, x)
            x = self.add_n([x, x])
            output.append(x)
        return output


class ListInputNetJit(NetJit):
    """
    Construct a multiple input network structure.
    """
    @jit(backend="ms_backend")
    def construct(self, list_x):
        output = []
        for i in range(g_block_num):
            x = self.add(list_x[i], list_x[i])
            x = self.add_n([x, x])
            x = self.mul(x, x)
            x = self.add_n([x, x])
            output.append(x)
        return output


class BaseNet(nn.Cell):
    def __init__(self, block_num=g_block_num):
        super().__init__()
        self.param = Parameter(Tensor(2, ms.float32))
        self.add = P.Add()
        self.mul = P.Mul()
        self.block_num = block_num

    def construct(self, x):
        output = []
        for _ in range(self.block_num):
            x = self.add(x, x)
            x = self.mul(x, 2.0)
            x = self.add(x, -0.1)
            x = self.mul(x, 0.6)
            output.append(x)
        return output


class BaseSeqNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.param = Parameter(Tensor(2, ms.float32))
        self.add = P.Add()
        self.net_jit = NetJit()
        self.net = BaseNet()
        self.add_n = P.AddN()

    def construct(self):
        pass


class BaseListInputSeqNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.param = Parameter(Tensor(2, ms.float32))
        self.add = P.Add()
        self.list_net_jit = ListInputNetJit()
        self.net = BaseNet()
        self.add_n = P.AddN()

    def construct(self):
        pass


class SingleInputWithDependOutput(BaseSeqNet):
    def construct(self, x):
        outputs = []
        origin_input = x
        out = self.net(x)
        for item in out:
            outputs.append(self.add(item, item))
        out_jit = self.net_jit(origin_input)
        for item in out_jit:
            outputs.append(self.add(item, item))

        for _ in range(inner_block_num):
            x = self.add_n(outputs)
            out_jit = self.net_jit(x)
            item = self.add_n(out_jit)
            outputs.append(self.add(item, item))

        output = self.add_n(outputs)
        return output


class SingleInputWithoutDepend(BaseSeqNet):
    def construct(self, x):
        outputs = []
        origin_input = x
        out = self.net(x)
        for item in out:
            outputs.append(self.add(item, item))
        out_jit = self.net_jit(origin_input)
        for item in out_jit:
            outputs.append(self.add(item, item))

        for _ in range(inner_block_num):
            out_jit = self.net_jit(origin_input)
            item = out_jit[0]
            outputs.append(self.add(item, item))


        output = self.add_n(outputs)
        return output

class SingleInputWithGraphOnly(BaseSeqNet):
    def construct(self, x):
        outputs = []
        origin_input = x
        out = self.net(x)
        for item in out:
            outputs.append(self.add(item, item))
        out_jit = self.net_jit(origin_input)
        for item in out_jit:
            outputs.append(self.add(item, item))

        for _ in range(inner_block_num):
            out_jit = self.net_jit(out_jit[0])

        item = out_jit[0]
        outputs.append(self.add(item, item))
        outputs.append(item)
        output = self.add_n(outputs)
        return output


class MultiInputWithDependOutput(BaseListInputSeqNet):
    def construct(self, x):
        outputs = []
        out = self.net(x)
        for item in out:
            outputs.append(self.add(item, item))

        for _ in range(inner_block_num):
            out = self.net(x)
            out = mutable(out)
            out_jit = self.list_net_jit(out)
            item = out_jit[0]
            outputs.append(self.add(item, item))

        output = self.add_n(outputs)
        return output


class MultiInputWithoutDepend(BaseListInputSeqNet):
    def construct(self, x):
        outputs = []
        out = self.net(x)
        for item in out:
            outputs.append(self.add(item, item))

        out = mutable(out)
        for _ in range(inner_block_num):
            out_jit = self.list_net_jit(out)
            item = out_jit[0]
            outputs.append(self.add(item, item))
            outputs.append(item)

        output = self.add_n(outputs)
        return output


class MultiInputGraphOnly(BaseListInputSeqNet):
    def construct(self, x):
        outputs = []
        out = self.net(x)
        for item in out:
            outputs.append(self.add(item, item))

        out_jit = out
        for _ in range(inner_block_num):
            out_jit = mutable(out_jit)
            out_jit = self.list_net_jit(out_jit)

        item = out_jit[0]
        outputs.append(self.add(item, item))
        outputs.append(item)
        output = self.add_n(outputs)
        return output


def execute_network_with_compare(input_data, static_net, dynamic_net):
    # warm up
    output = static_net(input_data)
    output[0].asnumpy()
    output = dynamic_net(input_data)
    output[0].asnumpy()

    # static shape network performance
    start_time = time.time()
    for _ in range(steps):
        static_output = static_net(input_data)
    static_output[0].asnumpy()
    end_time = time.time()
    static_net_cost_time = end_time - start_time

    # dynamic shape network performance
    start_time = time.time()
    for _ in range(steps):
        dynamic_output = dynamic_net(input_data)
    dynamic_output[0].asnumpy()
    end_time = time.time()
    dynamic_net_cost_time = end_time - start_time

    assert np.allclose(static_output[0].asnumpy(), dynamic_output[0].asnumpy())

    diff_time = abs(dynamic_net_cost_time - static_net_cost_time)
    print("performance diff ratio: ", diff_time / static_net_cost_time)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.0)
def test_async_pipe_dynamic_vs_static_shape_block_num_5():
    """
    Feature: Graph support async pipeline.
    Description: Test basic network performance and accuracy, dynamic shape vs static shape and block number is set to 5
    Expectation: The program execute and exit normally, the performance deterioration of dynamic shape network does not
                 exceed 10% compared to static shape network.
    """
    input_data = Tensor(np.zeros((32, 64)).astype(np.float32))
    dyn_input_data = Tensor(shape=(None, None), dtype=mstype.float32)

    block_num = 5
    static_net = BaseNet(block_num)
    dynamic_net = BaseNet(block_num)
    static_net.construct = jit(static_net.construct, backend='ms_backend')
    dynamic_net.construct = jit(dynamic_net.construct, backend='ms_backend')
    dynamic_net.set_inputs(dyn_input_data)

    execute_network_with_compare(input_data, static_net, dynamic_net)


@pytest.mark.skip(reason="The CI pipeline is not suitable for monitoring performance case,"
                  "removing post-test smoke testing phase.")
@arg_mark(plat_marks=['platform_ascend'], level_mark='level2', card_mark='onecard', essential_mark='essential')
def test_async_pipe_large_input_dynamic_vs_static_shape_block_num_50():
    """
    Feature: Graph support async pipeline.
    Description: Test basic network performance and accuracy with large input shape, dynamic shape vs static shape
                 and block number is set to 50
    Expectation: The program execute and exit normally, the performance deterioration of dynamic shape network does
                 not exceed 10% compared to static shape network.
    """
    input_data = Tensor(np.zeros((1024, 1024)).astype(np.float32))
    dyn_input_data = Tensor(shape=(None, None), dtype=mstype.float32)

    block_num = 50
    static_net = BaseNet(block_num)
    dynamic_net = BaseNet(block_num)
    static_net.construct = jit(static_net.construct, backend='ms_backend')
    dynamic_net.construct = jit(dynamic_net.construct, backend='ms_backend')
    dynamic_net.set_inputs(dyn_input_data)

    execute_network_with_compare(input_data, static_net, dynamic_net)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_async_pipe_single_input_multi_output_depend_op():
    """
    Feature: Graph support async pipeline.
    Description: Test single input, multiple output, subgraph input dependent on single operator output scenario.
    Expectation: The program execute and exit normally.
    """
    input_data = Tensor(np.zeros((2, 3)).astype(np.float32))

    net = SingleInputWithDependOutput()
    # warm up
    output = net(input_data)
    output = net(input_data)
    print(output)

    start_time = time.time()
    for _ in range(steps):
        output = net(input_data)
    output.asnumpy()
    end_time = time.time()
    print(output)
    cost_time = end_time - start_time
    print("Total time cost: ", cost_time, flush=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_async_pipe_single_input_multi_output_not_depend_op():
    """
    Feature: Graph support async pipeline.
    Description: Test scenario for single input, multiple outputs, where subgraphs do not
                 depend on the outputs of single operators.
    Expectation: The program execute and exit normally.
    """
    input_data = Tensor(np.zeros((2, 3)).astype(np.float32))

    net = SingleInputWithoutDepend()
    # warm up
    output = net(input_data)
    output = net(input_data)
    print(output)

    start_time = time.time()
    for _ in range(steps):
        output = net(input_data)
    output.asnumpy()
    end_time = time.time()
    print(output)
    cost_time = end_time - start_time
    print("Total time cost: ", cost_time, flush=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_async_pipe_single_input_multi_output_only_graph():
    """
    Feature: Graph support async pipeline.
    Description: Test for single input, multiple outputs, executing only forward subgraphs, without single operators.
    Expectation: The program execute and exit normally.
    """
    input_data = Tensor(np.zeros((2, 3)).astype(np.float32))

    net = SingleInputWithGraphOnly()
    # warm up
    output = net(input_data)
    output = net(input_data)
    print(output)

    start_time = time.time()
    for _ in range(steps):
        output = net(input_data)
    output.asnumpy()
    end_time = time.time()
    print(output)
    cost_time = end_time - start_time
    print("Total time cost: ", cost_time, flush=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.0)
def test_async_pipe_multi_input_multi_output_depend_op():
    """
    Feature: Graph support async pipeline.
    Description: Test multiple input, multiple output, graphs depend on single operator output scenario.
    Expectation: The program execute and exit normally.
    """
    input_data = Tensor(np.zeros((2, 3)).astype(np.float32))

    net = MultiInputWithDependOutput()
    # warm up
    output = net(input_data)
    output = net(input_data)
    print(output)

    start_time = time.time()
    for _ in range(steps):
        output = net(input_data)
    output.asnumpy()
    end_time = time.time()
    print(output)
    cost_time = end_time - start_time
    print("Total time cost: ", cost_time, flush=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_async_pipe_multi_input_multi_output_without_depend_op():
    """
    Feature: Graph support async pipeline.
    Description: Test multiple input, multiple output, graphs don't depend on single operator output scenario.
    Expectation: The program execute and exit normally.
    """
    input_data = Tensor(np.zeros((2, 3)).astype(np.float32))

    net = MultiInputWithoutDepend()
    # warm up
    output = net(input_data)
    output = net(input_data)
    print(output)

    start_time = time.time()
    for _ in range(steps):
        output = net(input_data)
    output.asnumpy()
    end_time = time.time()
    print(output)
    cost_time = end_time - start_time
    print("Total time cost: ", cost_time, flush=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_async_pipe_multi_input_multi_output_graph_only():
    """
    Feature: Graph support async pipeline.
    Description: Test multiple input, multiple output, only graph without single op case.
    Expectation: The program execute and exit normally.
    """
    input_data = Tensor(np.zeros((2, 3)).astype(np.float32))

    net = MultiInputGraphOnly()
    # warm up
    output = net(input_data)
    output = net(input_data)
    print(output)

    start_time = time.time()
    for _ in range(steps):
        output = net(input_data)
    output.asnumpy()
    end_time = time.time()
    print(output)
    cost_time = end_time - start_time
    print("Total time cost: ", cost_time, flush=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_async_pipe_single_input_multi_output_with_grad_depend_op():
    """
    Feature: Graph support async pipeline.
    Description: Test single input, multiple output, graph(forward+backward) input depends on
                 single operator output case.
    Expectation: The program execute and exit normally.
    """
    input_data = Tensor(np.zeros((5, 3)).astype(np.float32))

    net = SingleInputWithDependOutput()
    grad = P.GradOperation()
    grad_fn = grad(net)
    grad_out = grad_fn(input_data)
    print("First call grad output: ", grad_out)

    start_time = time.time()
    for _ in range(steps):
        grad_out = grad_fn(input_data)
    grad_out.asnumpy()
    end_time = time.time()
    print("Finally call grad output: ", grad_out)
    cost_time = end_time - start_time
    print("Total time cost: ", cost_time, flush=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_async_pipe_single_input_multi_output_with_grad():
    """
    Feature: Graph support async pipeline.
    Description: Test single input, multiple output, graph(forward+backward) input doesn't depend on single
                 operator output case.
    Expectation: The program execute and exit normally.
    """
    input_data = Tensor(np.zeros((5, 3)).astype(np.float32))

    net = SingleInputWithoutDepend()
    grad = P.GradOperation()
    grad_fn = grad(net)
    grad_out = grad_fn(input_data)

    print("First call grad output: ", grad_out)
    start_time = time.time()
    for _ in range(steps):
        grad_out = grad_fn(input_data)
    grad_out.asnumpy()
    end_time = time.time()
    print("Finally call grad output: ", grad_out)
    cost_time = end_time - start_time
    print("Total time cost: ", cost_time, flush=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_async_pipe_single_input_multi_output_only_graph_with_grad():
    """
    Feature: Graph support async pipeline.
    Description: Test single input, multiple output, only subgraph with grad case.
    Expectation: The program execute and exit normally.
    """
    input_data = Tensor(np.zeros((5, 3)).astype(np.float32))

    net = SingleInputWithGraphOnly()
    grad = P.GradOperation()
    grad_fn = grad(net)
    grad_out = grad_fn(input_data)
    print("First call grad output: ", grad_out)

    start_time = time.time()
    for _ in range(steps):
        grad_out = grad_fn(input_data)
    grad_out.asnumpy()
    end_time = time.time()
    print("Finally call grad output: ", grad_out)
    cost_time = end_time - start_time
    print("Total time cost: ", cost_time, flush=True)
