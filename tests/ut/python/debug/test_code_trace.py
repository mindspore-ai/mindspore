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
from mindspore.common.api import _cell_graph_executor, _MindsporeFunctionExecutor
from tests.security_utils import security_off_wrap
from tests.code_trace_analyzer import CodeTraceAnalyzer
from tests.ut.python.debug.resnet import resnet50, DatasetResNet

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
    _ms_function_executor = _MindsporeFunctionExecutor(func, int(time.time() * 1e9))
    x = Tensor(np.arange(10).reshape(10).astype(np.float32))
    y = Tensor(np.array([-1, 0, 1]).astype(np.float32))
    _ms_function_executor.compile("fn", x, y)

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
