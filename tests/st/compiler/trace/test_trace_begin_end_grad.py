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
""" test trace functions """

import numpy as np
import pytest
import mindspore as ms
from mindspore.ops.functional import grad
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.optim.momentum import Momentum
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.train import Model
from mindspore.graph.jit_begin_end import _jit_begin as jit_begin
from mindspore.graph.jit_begin_end import _jit_end as jit_end
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.common.parameter import Parameter, ParameterTuple
from tests.mark_utils import arg_mark
from tests.dataset_mock import MindData
from tests.st.networks.models.resnetv1_5 import ResidualBlock, conv7x7, bn_with_initialize, MakeLayer0, MakeLayer1, MakeLayer2, MakeLayer3, fc_with_initialize
from tests.st.networks.models.resnetv1_5 import resnet50 as resnet_pynative

grad_all = C.GradOperation(get_all=True)
grad_by_list = C.GradOperation(get_by_list=True)

@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_trace_begin_end_1():
    """
    Feature: JIT trace function
    Description: JIT trace function
    Expectation: No exception
    """
    class TraceNet(ms.nn.Cell):
        def __init__(self):
            super(TraceNet, self).__init__()
            self.x = ms.Tensor(1)

        def construct(self, x, y):
            jit_begin("__trace__jit_block__1__", x, y)
            a = ms.Tensor(2)
            z = x + a
            z = z + self.x
            z = z * y
            z = jit_end(z)
            return z

    class GradNet(ms.nn.Cell):
        def __init__(self):
            super(GradNet, self).__init__()
            self.net = TraceNet()

        def construct(self, x, y):
            z1 = x * y
            z2 = x + y
            z3 = self.net(z1, z2)
            return z3 * z3

    grad_net = GradNet()
    res1 = grad(grad_net)(ms.Tensor(1), ms.Tensor(3))
    res2 = grad(grad_net)(ms.Tensor(1), ms.Tensor(3))
    res3 = grad(grad_net)(ms.Tensor(1), ms.Tensor(3))
    res4 = grad(grad_net)(ms.Tensor(1), ms.Tensor(3))
    print(f'res1: {res1}, res2: {res2}, res3: {res3}, res4: {res4}')
    assert res1 == res2
    assert res2 == res3
    assert res3 == res4


GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0
clip_grad = C.MultitypeFuncGraph("clip_grad")
grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, gradient):
    return gradient * reciprocal(scale)


update_cell = DynamicLossScaleUpdateCell(
    loss_scale_value=65536, scale_factor=2, scale_window=1000)


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, gradient):
    dt = F.dtype(gradient)
    if clip_type == 0:
        new_grad = ops.clip_by_value(gradient, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                     F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(gradient, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


class TrainOneStepWithLossScaleCell(nn.Cell):
    def __init__(self, network, optimizer, scale_update_cell=None):
        super(TrainOneStepWithLossScaleCell, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=True)
        self.reducer_flag = False
        self.grad_reducer = nn.Identity()
        self.cast = P.Cast()
        self.alloc_status = P.NPUAllocFloatStatus()
        self.get_status = P.NPUGetFloatStatus()
        self.clear_status = P.NPUClearFloatStatus()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.hyper_map = C.HyperMap()
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32),
                                        name="loss_scale")

    def construct(self, x, sens=None):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(x)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens
        # alloc status and clear should be right before gradoperation
        init = self.alloc_status()
        init = F.depend(init, loss)
        clear_status = self.clear_status(init)
        scaling_sens = F.depend(scaling_sens, clear_status)
        grads = self.grad(self.network, weights)(
            x, self.cast(scaling_sens, mstype.float32))
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(
            F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        init = F.depend(init, grads)
        get_status = self.get_status(init)
        init = F.depend(init, get_status)
        flag_sum = self.reduce_sum(init, (0,))
        cond = self.less_equal(self.base, flag_sum)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if not overflow:
            self.optimizer(grads)
        return (loss, cond, scaling_sens.value())


class DatasetLenet(MindData):
    def __init__(self, predict, label, length=3):
        super(DatasetLenet, self).__init__(size=length)
        self.predict = predict
        self.label = label
        self.index = 0
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        return self.predict, self.label

    def reset(self):
        self.index = 0


class LoopLayer(nn.Cell):
    def __init__(self):
        super(LoopLayer, self).__init__()
        self.matmul = P.MatMul()
        self.relu = P.ReLU()
        self.matmul_weight = Parameter(
            Tensor(np.ones([64, 64]), dtype=ms.float32), name="weight")

    def construct(self, x):
        out = self.matmul(x, self.matmul_weight)
        out = self.relu(out)
        return out


class Net1(nn.Cell):
    def __init__(self):
        super(Net1, self).__init__()
        self.exp = P.Exp()
        self.mean = P.ReduceMean()
        layers = []
        for _ in range(3):
            layer = LoopLayer()
            layers.append(layer)
        self.layers = nn.CellList(layers)

    def construct(self, x):
        jit_begin("__trace__jit_block__1__", x)
        out = self.exp(x)
        for layer in self.layers:
            layer_out = layer(out)
            out = layer_out
        out = self.mean(out, -1)
        out = jit_end(out)
        return out


class Net2(nn.Cell):
    def __init__(self):
        super(Net2, self).__init__()
        self.exp = P.Exp()
        self.mean = P.ReduceMean()
        layers = []
        for _ in range(3):
            layer = LoopLayer()
            layers.append(layer)
        self.layers = nn.CellList(layers)

    def construct(self, x):
        out = self.exp(x)
        for layer in self.layers:
            jit_begin("__trace__jit_block__1__", out)
            layer_out = layer(out)
            out = layer_out
            out = jit_end(out)
        out = self.mean(out, -1)
        return out


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
@pytest.mark.skip(reason="skip case since weight with same name will throw error for now.")
def test_trace_begin_end_train_1():
    """
    Feature: JIT trace function
    Description: JIT trace function on model train
    Expectation: No exception
    """
    predict = Tensor(np.ones([64, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64,]), dtype=ms.int32)
    dataset = DatasetLenet(predict, label)
    net = Net1()
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)
    net = TrainOneStepWithLossScaleCell(net, opt, update_cell)
    model = Model(network=net)
    model.train(2, dataset, dataset_sink_mode=False)
    assert [x for x in net.get_parameters()][0].value()[0][0] != 1


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
@pytest.mark.skip(reason="skip case since weight with same name will throw error for now.")
def test_trace_begin_end_train_2():
    """
    Feature: JIT trace function
    Description: JIT trace function on model train
    Expectation: No exception
    """
    predict = Tensor(np.ones([64, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64,]), dtype=ms.int32)
    dataset = DatasetLenet(predict, label)
    net = Net1()
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)
    net = TrainOneStepWithLossScaleCell(net, opt, update_cell)
    model = Model(network=net)
    model.train(2, dataset, dataset_sink_mode=False)
    assert [x for x in net.get_parameters()][0].value()[0][0] != 1


class ResNet(nn.Cell):

    def __init__(self, block, num_classes=100, batch_size=32):
        super(ResNet, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.conv1 = conv7x7(3, 64, stride=2, padding=0)

        self.bn1 = bn_with_initialize(64)
        self.relu = P.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="SAME")

        self.layer1 = MakeLayer0(
            block, in_channels=64, out_channels=256, stride=1)
        self.layer2 = MakeLayer1(
            block, in_channels=256, out_channels=512, stride=2)
        self.layer3 = MakeLayer2(
            block, in_channels=512, out_channels=1024, stride=2)
        self.layer4 = MakeLayer3(
            block, in_channels=1024, out_channels=2048, stride=2)

        self.pool = P.ReduceMean(keep_dims=True)
        self.fc = fc_with_initialize(512 * block.expansion, num_classes)
        self.flatten = nn.Flatten()

    def construct(self, x):
        jit_begin("__trace__jit_block__1__", x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x, (-2, -1))
        x = self.flatten(x)
        x = self.fc(x)
        x = jit_end(x)
        return x


def resnet50(batch_size, num_classes):
    return ResNet(ResidualBlock, num_classes, batch_size)


def train(net, data, label):
    learning_rate = 0.01
    momentum = 0.9

    optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate, momentum)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(
        net_with_criterion, optimizer)  # optimizer
    train_network.set_train()
    res = train_network(data, label)
    return res


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
@pytest.mark.skip(reason="skip case since weight with same name will throw error for now.")
def test_resnet50_with_trace():
    """
    Feature: JIT trace function
    Description: JIT trace function
    Expectation: No exception
    """
    data = Tensor(np.ones([32, 3, 224, 224]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([32]).astype(np.int32))
    net = resnet50(32, 10)
    net_pynative = resnet_pynative(32, 10)
    res_trace = train(net, data, label)
    res_pynative = train(net_pynative, data, label)
    assert np.allclose(res_trace.asnumpy(), res_pynative.asnumpy())


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_control_with_trace():
    """
    Feature: Switch simplify pass.
    Description: If switch simplify pass can't simplify constant tensor condition,
                 dead node will exist in backend.
    Expectation: output correct.
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.op = P.Add()

        def construct(self, x, y):
            if y != x:
                x = y - 3
            elif x == 4:
                for r in range(2):
                    if x > 2:
                        jit_begin("__trace__jit_block__1__", x, y)
                        y = y + 3
                        y = y - y
                        y = y * x
                        y = jit_end(y)
                    elif y >= x:
                        x = x * x
                    elif x > y:
                        x = y - r
                    else:
                        y = 2 + x
                    for _ in range(2):
                        jit_begin("__trace__jit_block__2__", x, y)
                        x = x * y
                        x = x - 3
                        y = y + 2
                        x, y = jit_end(x, y)
                        if x > 3:
                            break
            elif x == y:
                if y <= x:
                    y = x / 2
                    x = 3 + y
                    x = x * 2
                elif x == 2:
                    x = y * y
                elif x < y:
                    y = 2 * y
                elif x != 2:
                    y = x * y
            while x != 5:
                break
            return self.op(x, y)

    x = np.array([4], np.float32)
    y = np.array([4], np.float32)
    net = Net()
    out = net(Tensor(x), Tensor(y))
    grad_net = F.grad(net, grad_position=(0, 1))
    fgrad = grad_net(Tensor(x), Tensor(y))
    assert np.allclose(out.asnumpy(), np.array([327], np.float32))
    assert np.allclose(fgrad[0].asnumpy(), np.array([0.], np.float32))
    assert np.allclose(fgrad[1].asnumpy(), np.array([0.], np.float32))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
@pytest.mark.skip(reason="skip case since weight with same name will throw error for now.")
def test_trace_while_grad():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    class MyWhileNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.max = P.ReduceMax()

        def construct(self, idx, end, x):
            while idx < end:
                jit_begin("__trace__jit_block__1__", idx, x)
                part = x[idx, :, :]
                max_num = self.max(part)
                x[idx, :, 0:2] = max_num
                idx = idx + 1
                idx, x = jit_end(idx, x)
            return x

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    idx = Tensor(np.array(0), dtype=ms.int32)
    end = Tensor(np.array(2), dtype=ms.int32)
    input_x = np.array([[[4, 0], [0, 0]],
                        [[0, 4], [0, 0]]]).astype(np.float32)
    x = Tensor(input_x, dtype=ms.float32)
    while_net = MyWhileNet()
    net = GradNet(while_net)
    graph_output = net(idx, end, x)

    expect_zero = np.array([0], dtype=np.float32)
    expect_two = input_x
    assert np.allclose(graph_output[0].asnumpy(), expect_zero, 0.0001, 0.0001)
    assert np.allclose(graph_output[1].asnumpy(), expect_zero, 0.0001, 0.0001)
    assert np.allclose(graph_output[2].asnumpy(), expect_two, 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='essential')
@pytest.mark.skip(reason="skip case since weight with same name will throw error for now.")
def test_while_with_param_grad():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    class MyWhileNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.max = P.ReduceMax()
            self.param = Parameter(
                Tensor(np.arange(2 * 2 * 2).reshape((2, 2, 2)), ms.float32), name="weight")
            self.zero = Tensor(np.zeros(([2, 2, 2])), ms.float32)

        def construct(self, idx, end, x):
            out = self.zero
            while idx < end:
                jit_begin("__trace__jit_block__1__", idx, x, out)
                part = x[idx, :, :]
                max_num = self.max(part)
                x[idx, :, 0:2] = max_num
                out = out + x + self.param
                idx = idx + 1
                idx, x, out = jit_end(idx, x, out)
            return out

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = ParameterTuple(net.trainable_params())

        def construct(self, a, b, c):
            return grad_by_list(self.net, self.weights)(a, b, c)

    while_net = MyWhileNet()
    net = GradNet(while_net)
    idx = Tensor(np.array(0), dtype=ms.int32)
    end = Tensor(np.array(2), dtype=ms.int32)
    x = Tensor(np.arange(8).reshape(2, 2, 2).astype(
        np.float32), dtype=ms.float32)
    graph_output = net(idx, end, x)
    expect = np.array([[[2, 2], [2, 2]], [[2, 2], [2, 2]]], dtype=np.int32)
    assert np.allclose(graph_output[0].asnumpy(), expect, 0.0001, 0.0001)
