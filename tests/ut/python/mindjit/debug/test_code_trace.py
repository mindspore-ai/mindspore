# Copyright 2023 Huawei Technologies Co., Ltd
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
import shutil
import pytest
import numpy as np
import mindspore
from mindspore import nn, Tensor, ops, context, jit, Model
from mindspore.nn import Cell
from mindspore.nn.optim.momentum import Momentum
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.graph.api import _cell_graph_executor, _JitExecutor
from mindspore.common.parameter import Parameter
from mindspore import dataset as ds
from tests.security_utils import security_off_wrap
from tests.code_trace_analyzer import CodeTraceAnalyzer
from tests.ut.python.mindjit.debug.resnet import resnet50, DatasetResNet

context.set_context(mode=context.GRAPH_MODE)


@security_off_wrap
def test_lenet_code_trace():
    """
    Feature: Code Trace.
    Description: Test Lenet code trace.
    Expectation: success.
    """

    class LeNet5(nn.Cell):
        def __init__(self, num_class=10, num_channel=1):
            super(LeNet5, self).__init__()
            self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
            self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
            self.relu = nn.ReLU()
            self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Dense(16 * 5 * 5, 120)
            self.fc2 = nn.Dense(120, 84)
            self.fc3 = nn.Dense(84, num_class)

        def construct(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.max_pool2d(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.max_pool2d(x)
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    save_graph_path = "test_lenet_code_trace"
    context.set_context(save_graphs=1, save_graphs_path=save_graph_path)
    net = LeNet5()
    input_tensor = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32) * 0.01)
    _cell_graph_executor.compile(net, input_tensor)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")
    accuracy = analyzer.analyze()
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)


@security_off_wrap
def test_resnet50_code_trace():
    """
    Feature: Code Trace.
    Description: Test ResNet50 code trace.
    Expectation: success.
    """

    save_graph_path = "test_resnet50_code_trace"
    context.set_context(save_graphs=1, save_graphs_path=save_graph_path)
    predict = Tensor(np.ones([32, 3, 224, 224]), dtype=mindspore.float32)
    label = Tensor(np.ones([32]), dtype=mindspore.int32)
    dataset = DatasetResNet(predict, label, 2)

    net = resnet50()
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.1, 0.9)
    model = Model(net, loss, opt)
    model.train(1, dataset, dataset_sink_mode=False)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")
    accuracy = analyzer.analyze()
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)


@security_off_wrap
def test_resnet50_node_fullname():
    """
    Feature: Code Trace.
    Description: Test ResNet50 code trace.
    Expectation: success.
    """

    save_graph_path = "test_resnet50_node_fullname"
    context.set_context(save_graphs=1, save_graphs_path=save_graph_path)
    predict = Tensor(np.ones([32, 3, 224, 224]), dtype=mindspore.float32)
    label = Tensor(np.ones([32]), dtype=mindspore.int32)
    dataset = DatasetResNet(predict, label, 2)

    net = resnet50()
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.1, 0.9)
    model = Model(net, loss, opt)
    model.train(1, dataset, dataset_sink_mode=False)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")
    res = analyzer.check_fullname("ApplyMomentum-op", 160)
    if not res:
        raise ValueError("fullname of ApplyMomentum is changed by ir print")

    shutil.rmtree(save_graph_path)


@security_off_wrap
def test_code_trace1():
    """
    Feature: Code Trace.
    Description: Test source code location.
    Expectation: success.
    """

    class Net(nn.Cell):
        def construct(self, x, y, z):
            res = ops.maximum(x, y)
            res = ops.maximum(res, z)
            return res

    save_graph_path = "test_code_trace1"
    context.set_context(save_graphs=1, save_graphs_path=save_graph_path)
    net = Net()
    x = Tensor([1], mindspore.float32)
    y = Tensor([2], mindspore.float32)
    z = Tensor([3], mindspore.float32)
    _cell_graph_executor.compile(net, x, y, z)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")
    accuracy = analyzer.analyze()
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)


@security_off_wrap
def test_code_trace2():
    """
    Feature: Code Trace.
    Description: Test source code location.
    Expectation: success.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.relu = nn.ReLU()

        def construct(self, x):
            x = self.relu(x)
            x = self.relu(x)
            return x

    save_graph_path = "test_code_trace2"
    context.set_context(save_graphs=True, save_graphs_path=save_graph_path)
    net = Net()
    x = Tensor([1], mindspore.float32)
    _cell_graph_executor.compile(net, x)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")
    accuracy = analyzer.analyze()
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)


@security_off_wrap
def test_code_trace3():
    """
    Feature: Code Trace.
    Description: Test source code location.
    Expectation: success.
    """

    @jit
    def func(x, y):
        x = x.sum(-1)
        y = y.sum()
        return x + y

    save_graph_path = "test_code_trace3"
    context.set_context(save_graphs=True, save_graphs_path=save_graph_path)
    _jit_executor = _JitExecutor(func, int(time.time() * 1e9))
    x = Tensor(np.arange(10).reshape(10).astype(np.float32))
    y = Tensor(np.array([-1, 0, 1]).astype(np.float32))
    _jit_executor.compile("fn", x, y)

    analyzer = CodeTraceAnalyzer(func, save_graph_path, "validate")
    accuracy = analyzer.analyze()
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)


@pytest.mark.skip(reason="'x = self.dense1(x)' not in ir")
@security_off_wrap
def test_code_trace4():
    """
    Feature: Code Trace.
    Description: Test source code location.
    Expectation: success.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.dense1 = nn.Dense(3, 4)
            self.dense2 = nn.Dense(4, 5)

        def construct(self, x):
            x = self.dense1(x)
            x = self.dense2(x)
            return x

    save_graph_path = "test_code_trace4"
    context.set_context(save_graphs=True, save_graphs_path=save_graph_path)
    net = Net()
    x = Tensor(np.array([[180, 234, 154], [244, 48, 247]]), mindspore.float32)
    _cell_graph_executor.compile(net, x)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")
    accuracy = analyzer.analyze()
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)


@security_off_wrap
def test_code_trace5():
    """
    Feature: Code Trace.
    Description: Test source code location.
    Expectation: success.
    """

    class Net(nn.Cell):
        def construct(self, x):
            x = x[0, 1]
            x = x[2, 3]
            return x

    save_graph_path = "test_code_trace5"
    context.set_context(save_graphs=True, save_graphs_path=save_graph_path)
    net = Net()
    x = Tensor(np.ones([1, 120, 1024, 640]), mindspore.float32)
    _cell_graph_executor.compile(net, x)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")
    accuracy = analyzer.analyze()
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)


@security_off_wrap
def test_code_trace_loop_stack_depth():
    """
    Feature: Code Trace.
    Description: Test stack depth.
    Expectation: success.
    """
    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.default = 4000

        def construct(self, x):
            output = x
            for i in range(self.default):
                if self.default is None:
                    output = output + i
            if output > self.default:
                return self.default
            return output

    with pytest.raises(RuntimeError):
        context.set_context(mode=context.GRAPH_MODE)
        net = Net()
        x = Tensor([-1])
        out = net(x)
        assert out == -1


class Net1(nn.Cell):
    def construct(self, x1, x2):
        out = x1 + x2
        return out


class Net2(nn.Cell):
    def construct(self, x3, x4):
        out = x3 + x4
        return out


class Net3(nn.Cell):
    def __init__(self):
        super(Net3, self).__init__()
        self.net1 = Net1()
        self.net2 = Net2()

    def construct(self, x1, x2, x3):
        out1 = self.net1(x1, x2)
        out2 = self.net2(out1, x3)
        return out2


def add(x, y):
    return x + y

def mul(x, y):
    return x * y


def div(x, y):
    return x / y


class Net4(nn.Cell):
    def construct(self, x3, x4):
        out = add(x3, x4)
        return out


class Net5(nn.Cell):
    def __init__(self):
        super(Net5, self).__init__()
        self.net1 = Net4()
        self.net2 = Net2()

    def construct(self, x1, x2, x3):
        out1 = self.net1(x1, x2)
        out2 = self.net2(out1, x3)
        return out2


class Net6(nn.Cell):
    def __init__(self):
        super(Net6, self).__init__()
        self.net1 = Net2()
        self.net2 = Net4()

    def construct(self, x1, x2, x3):
        out1 = self.net1(x1, x2)
        out2 = self.net2(out1, x3)
        return out2


class Net7(nn.Cell):
    def __init__(self):
        super(Net7, self).__init__()
        self.net = Net4()
    def construct(self, x, y):
        out = self.net(x, y)
        return out

class Net8(nn.Cell):
    def __init__(self):
        super(Net8, self).__init__()
        self.net1 = Net7()
        self.net2 = Net4()

    def construct(self, x1, x2, x3):
        out1 = self.net1(x1, x2)
        out2 = self.net2(out1, x3)
        return out2


def cal(x, y):
    mul_res = mul(x, y)
    mul_res2 = mul(mul_res, x)
    div_res = div(mul_res2, y)
    out = add(div_res, mul_res)
    return out


class Net9(nn.Cell):
    def construct(self, x3, x4):
        out = add(x3, x4)
        return out


class Net10(nn.Cell):
    def construct(self, x3, x4):
        mul_res = mul(x3, x4)
        mul_res2 = mul(mul_res, x3)
        div_res = div(mul_res2, x4)
        out = add(div_res, mul_res)
        return out


class Net11(nn.Cell):
    def __init__(self):
        super(Net11, self).__init__()
        self.net1 = Net9()
        self.net2 = Net10()

    def construct(self, x1, x2, x3):
        out1 = self.net1(x1, x2)
        out2 = self.net2(out1, x3)
        return out2


@security_off_wrap
def test_code_trace6():
    """
    Feature: Code Trace.
    Description: Test source code location.
    Expectation: success.
    """

    save_graph_path = "test_code_trace6"
    context.set_context(save_graphs=True, save_graphs_path=save_graph_path)
    net = Net3()
    x1 = Tensor([[1, 2], [3, 4]])
    x2 = Tensor([[1.0, 2.0], [3.0, 4.0]])
    _cell_graph_executor.compile(net, x1, x1, x2)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")

    # return lines of sub cells will not be counted since those cells are inlined
    accuracy = analyzer.analyze(2)
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)


@security_off_wrap
def test_code_trace7():
    """
    Feature: Code Trace.
    Description: Test source code location.
    Expectation: success.
    """

    save_graph_path = "test_code_trace7"
    context.set_context(save_graphs=True, save_graphs_path=save_graph_path)
    net = Net3()
    x1 = Tensor([[1, 2], [3, 4]])
    x2 = Tensor([[1, 2], [3, 4]])
    _cell_graph_executor.compile(net, x1, x1, x2)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")

    # return lines of sub cells will not be counted since those cells are inlined
    accuracy = analyzer.analyze(2)
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)


@security_off_wrap
def test_code_trace8():
    """
    Feature: Code Trace.
    Description: Test source code location.
    Expectation: success.
    """

    class Net_Inner(nn.Cell):
        def construct(self, x1, x2, x3):
            out1 = x1 + x2
            out2 = out1 + x3
            return out2

    save_graph_path = "test_code_trace8"
    context.set_context(save_graphs=True, save_graphs_path=save_graph_path)
    net = Net_Inner()
    x1 = Tensor([[1, 2], [3, 4]])
    x2 = Tensor([[1.0, 2.0], [3.0, 4.0]])
    _cell_graph_executor.compile(net, x1, x1, x2)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")
    accuracy = analyzer.analyze()
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)


@security_off_wrap
def test_code_trace9():
    """
    Feature: Code Trace.
    Description: Test source code location.
    Expectation: success.
    """

    save_graph_path = "test_code_trace9"
    context.set_context(save_graphs=True, save_graphs_path=save_graph_path)
    net = Net5()
    x1 = Tensor([[1, 2], [3, 4]])
    x2 = Tensor([[1.0, 2.0], [3.0, 4.0]])
    _cell_graph_executor.compile(net, x1, x1, x2)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")

    # return lines of sub cells will not be counted since those cells are inlined
    accuracy = analyzer.analyze(2)
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)


@security_off_wrap
def test_code_trace10():
    """
    Feature: Code Trace.
    Description: Test source code location.
    Expectation: success.
    """

    save_graph_path = "test_code_trace10"
    context.set_context(save_graphs=True, save_graphs_path=save_graph_path)
    net = Net6()
    x1 = Tensor([[1, 2], [3, 4]])
    x2 = Tensor([[1.0, 2.0], [3.0, 4.0]])
    _cell_graph_executor.compile(net, x1, x1, x2)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")

    # return lines of sub cells will not be counted since those cells are inlined
    accuracy = analyzer.analyze(2)
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)


@security_off_wrap
def test_code_trace11():
    """
    Feature: Code Trace.
    Description: Test source code location.
    Expectation: success.
    """

    save_graph_path = "test_code_trace11"
    context.set_context(save_graphs=True, save_graphs_path=save_graph_path)
    net = Net8()
    x1 = Tensor([[1, 2], [3, 4]])
    x2 = Tensor([[1.0, 2.0], [3.0, 4.0]])
    _cell_graph_executor.compile(net, x1, x1, x2)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")

    # return lines of sub cells will not be counted since those cells are inlined
    accuracy = analyzer.analyze(3)
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)


@security_off_wrap
def test_code_trace12():
    """
    Feature: Code Trace.
    Description: Test source code location.
    Expectation: success.
    """

    save_graph_path = "test_code_trace12"
    context.set_context(save_graphs=True, save_graphs_path=save_graph_path)
    net = Net11()
    x1 = Tensor([[1, 2], [3, 4]])
    x2 = Tensor([[1.0, 2.0], [3.0, 4.0]])
    _cell_graph_executor.compile(net, x1, x1, x2)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")

    # return lines of sub cells will not be counted since those cells are inlined
    accuracy = analyzer.analyze(2)
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)


class AddNet(nn.Cell):
    def __init__(self, weight_shape, dtype=mindspore.float32):
        super(AddNet, self).__init__()
        self.add = ops.Add()
        self.add_weight = Parameter(
            Tensor(np.full(weight_shape, 2.0), dtype=dtype), name="matmul_weight")

    def construct(self, x, y):
        out = x + y
        out = self.add(out, self.add_weight)
        out = out + out
        return out


def generator(size, dtype=mindspore.float32, label_dtype=mindspore.float32):
    for _ in range(2):
        inputs, label = Tensor(np.full(size[0], 0.5), dtype=dtype), Tensor(np.full(size[1], 0.5),
                                                                           dtype=label_dtype)
        yield inputs, label


@security_off_wrap
def test_code_trace13():
    """
    Feature: Code Trace.
    Description: Test source code location.
    Expectation: success.
    """

    save_graph_path = "test_code_trace13"
    context.set_context(save_graphs=True, save_graphs_path=save_graph_path)
    net = AddNet(weight_shape=(1, 2), dtype=mindspore.float32)


    opt_fn = Momentum(learning_rate=0.01, momentum=0.9,
                      params=net.get_parameters())
    dataset = ds.GeneratorDataset(lambda: generator(size=((2, 2), (2, 2)), label_dtype=mindspore.float32),
                                  ["inputs", "label"])
    model = Model(net, optimizer=opt_fn)
    model.train(2, dataset, dataset_sink_mode=False)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")

    # return lines of sub cells will not be counted since those cells are inlined
    accuracy = analyzer.analyze(1)
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)


@security_off_wrap
def test_code_trace14():
    """
    Feature: Code Trace.
    Description: Test source code location when there is a line with 4096 chars.
    Expectation: success.
    """

    class Net(nn.Cell):
        def construct(self, x, y, z):
            #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
            res = ops.maximum(x, y)
            res = ops.maximum(res, z)
            return res

    save_graph_path = "test_code_trace14"
    context.set_context(save_graphs=1, save_graphs_path=save_graph_path)
    net = Net()
    x = Tensor([1], mindspore.float32)
    y = Tensor([2], mindspore.float32)
    z = Tensor([3], mindspore.float32)
    _cell_graph_executor.compile(net, x, y, z)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")
    accuracy = analyzer.analyze()
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)


@security_off_wrap
def test_code_trace15():
    """
    Feature: Code Trace.
    Description: Test source code location when there are two lines with over 4096 chars.
    Expectation: success.
    """

    class Net(nn.Cell):
        def construct(self, x, y, z):
            #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
            #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
            res = ops.maximum(x, y)
            res = ops.maximum(res, z)
            return res

    save_graph_path = "test_code_trace15"
    context.set_context(save_graphs=1, save_graphs_path=save_graph_path)
    net = Net()
    x = Tensor([1], mindspore.float32)
    y = Tensor([2], mindspore.float32)
    z = Tensor([3], mindspore.float32)
    _cell_graph_executor.compile(net, x, y, z)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")
    accuracy = analyzer.analyze()
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)
