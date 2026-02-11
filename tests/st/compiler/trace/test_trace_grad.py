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
""" test trace functions """

import shutil
import os
import numpy as np
import pytest
from tests.mark_utils import arg_mark
from tests.dataset_mock import MindData
import mindspore as ms
from mindspore.ops.functional import grad
from mindspore import Tensor, context, jit
from mindspore.common.parameter import Parameter
from mindspore.common import dtype as mstype
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.optim.momentum import Momentum
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.train import Model
from mindspore import ParameterTuple


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='essential')
def test_trace_1():
    """
    Feature: JIT trace function
    Description: JIT trace function
    Expectation: No exception
    """
    class TraceNet(ms.nn.Cell):
        def __init__(self):
            super(TraceNet, self).__init__()
            self.x = ms.Tensor(1)

        @ms.jit(capture_mode="trace")
        def construct(self, x, y):
            a = ms.Tensor(2)
            z = x + a
            z = z + self.x
            z = z * y
            return z

    trace_net = TraceNet()
    res1 = grad(trace_net)(ms.Tensor(1), ms.Tensor(3))
    res2 = grad(trace_net)(ms.Tensor(1), ms.Tensor(3))
    print(f'res1: {res1}, res2: {res2}')
    assert res1 == res2


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_trace_2():
    """
    Feature: JIT trace function
    Description: JIT trace function
    Expectation: No exception
    """
    class TraceNet(ms.nn.Cell):
        def __init__(self):
            super(TraceNet, self).__init__()
            self.x = ms.Tensor(1)

        @ms.jit(capture_mode="trace")
        def construct(self, x, y):
            a = ms.Tensor(2)
            z = x + a
            z = z + self.x
            z = z * y
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


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='essential')
def test_trace_3():
    """
    Feature: JIT trace function
    Description: JIT trace function
    Expectation: No exception
    """
    class TraceNet(ms.nn.Cell):
        def __init__(self):
            super(TraceNet, self).__init__()
            self.x = ms.Tensor(1)

        @ms.jit(capture_mode="trace")
        def construct(self, x, y):
            a = ms.Tensor(2)
            z = x + a
            z = z + self.x
            z = z * y
            return z

    trace_net = TraceNet()
    forward_res = trace_net(ms.Tensor(1), ms.Tensor(3))
    res1 = grad(trace_net)(ms.Tensor(1), ms.Tensor(3))
    res2 = grad(trace_net)(ms.Tensor(1), ms.Tensor(3))
    print(f'res1: {res1}, res2: {res2}, forward_res: {forward_res}')
    assert res1 == res2


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


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.exp = P.Exp()
        self.mean = P.ReduceMean()
        layers = []
        for _ in range(3):
            layer = LoopLayer()
            layers.append(layer)
        self.layers = nn.CellList(layers)

    @ms.jit(capture_mode="trace")
    def construct(self, x):
        out = self.exp(x)
        for layer in self.layers:
            layer_out = layer(out)
            out = layer_out
        out = self.mean(out, -1)
        return out


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
@pytest.mark.skip(reason="skip")
def test_trace_train_1():
    """
    Feature: JIT trace function
    Description: JIT trace function on model train
    Expectation: No exception
    """
    predict = Tensor(np.ones([64, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64,]), dtype=ms.int32)
    dataset = DatasetLenet(predict, label)
    net = Net()
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)
    net = TrainOneStepWithLossScaleCell(net, opt, update_cell)
    model = Model(network=net)
    model.train(2, dataset, dataset_sink_mode=False)
    assert [x for x in net.get_parameters()][0].value()[0][0] != 1


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
@pytest.mark.skip(reason="skip")
def test_trace_train_2():
    """
    Feature: JIT trace function
    Description: JIT trace function on model train
    Expectation: No exception
    """
    context.set_context(save_graphs=True, save_graphs_path="./ir")
    predict = Tensor(np.ones([64, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64,]), dtype=ms.int32)
    dataset = DatasetLenet(predict, label)
    net = Net()
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)
    net = TrainOneStepWithLossScaleCell(net, opt, update_cell)
    model = Model(network=net)
    model.train(2, dataset, dataset_sink_mode=False)
    if os.path.exists("./ir"):
        shutil.rmtree("./ir")
    context.set_context(save_graphs=False)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_trace_4():
    """
    Feature: JIT trace function
    Description: JIT trace function
    Expectation: No exception
    """
    class TraceNet(ms.nn.Cell):
        def __init__(self):
            super(TraceNet, self).__init__()
            self.x = ms.Tensor(1)

        @ms.jit(capture_mode="trace")
        def construct(self, x):
            a = ms.Tensor(2)
            z = x[0] + a
            z = z + self.x
            z = z * x[1]
            return z

    class JitNet(ms.nn.Cell):
        def __init__(self):
            super(JitNet, self).__init__()
            self.x = ms.Tensor(1)

        @jit
        def construct(self, x):
            a = ms.Tensor(2)
            z = x[0] + a
            z = z + self.x
            z = z * x[1]
            return z

    trace_net = TraceNet()
    jit_net = JitNet()
    res1 = grad(trace_net)((ms.Tensor(1), ms.Tensor(3)))
    res2 = grad(jit_net)((ms.Tensor(1), ms.Tensor(3)))
    print(f'res1: {res1}, res2: {res2}')
    assert res1 == res2


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_trace_5():
    """
    Feature: JIT trace function
    Description: JIT trace function
    Expectation: No exception
    """
    class TraceNet(ms.nn.Cell):
        def __init__(self):
            super(TraceNet, self).__init__()
            self.x = ms.Tensor(1)

        @ms.jit(capture_mode="trace")
        def construct(self, x, y):
            a = ms.Tensor(2)
            z = x + a
            z = z + self.x
            z = z * y
            return z

    class JitNet(ms.nn.Cell):
        def __init__(self):
            super(JitNet, self).__init__()
            self.x = ms.Tensor(1)

        @jit
        def construct(self, x, y):
            a = ms.Tensor(2)
            z = x + a
            z = z + self.x
            z = z * y
            return z

    class GradNet(ms.nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, x, y):
            z1 = x * y
            z2 = x + y
            z3 = self.net(z1, z2)
            return z3 * z3

    trace_net = TraceNet()
    jit_net = JitNet()
    trace_grad_net = GradNet(trace_net)
    jit_grad_net = GradNet(jit_net)
    res1 = grad(trace_grad_net)(ms.Tensor(1), ms.Tensor(3))
    res2 = grad(trace_grad_net)(ms.Tensor(1), ms.Tensor(3))
    res3 = grad(trace_grad_net)(ms.Tensor(1), ms.Tensor(3))
    res4 = grad(jit_grad_net)(ms.Tensor(1), ms.Tensor(3))
    print(f'res1: {res1}, res2: {res2}, res3: {res3}, res4: {res4}')
    assert res1 == res2
    assert res2 == res3
    assert res3 == res4


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_trace_6():
    """
    Feature: JIT trace function
    Description: JIT trace function with pyboost ops
    Expectation: No exception
    """
    class TraceNet(ms.nn.Cell):
        def __init__(self):
            super(TraceNet, self).__init__()
            self.x = ms.Tensor([5.0, 6.0])

        @ms.jit(capture_mode="trace")
        def construct(self, x, y):
            a = ms.Tensor([7.0, 8.0])
            z = x + a
            z = z.argmax()
            z = z - self.x
            z = z.tanh()
            z = -z
            z = z @ y

            return z

    class JitNet(ms.nn.Cell):
        def __init__(self):
            super(JitNet, self).__init__()
            self.x = ms.Tensor([5.0, 6.0])

        @jit
        def construct(self, x, y):
            a = ms.Tensor([7.0, 8.0])
            z = x + a
            z = z.argmax()
            z = z - self.x
            z = z.tanh()
            z = -z
            z = z @ y
            return z

    class GradNet(ms.nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, x, y):
            z1 = x * y
            z2 = x + y
            z3 = self.net(z1, z2)
            return z3 * z3

    trace_net = TraceNet()
    jit_net = JitNet()
    trace_grad_net = GradNet(trace_net)
    jit_grad_net = GradNet(jit_net)
    res1 = grad(trace_grad_net)(ms.Tensor([1.0, 2.0]), ms.Tensor([3.0, 4.0]))
    res2 = grad(jit_grad_net)(ms.Tensor([1.0, 2.0]), ms.Tensor([3.0, 4.0]))
    print(f'res1: {res1}, res2: {res2}')
    assert np.allclose(res1.asnumpy(), res2.asnumpy())


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_trace_7():
    """
    Feature: JIT trace function
    Description: JIT trace function with pyboost ops
    Expectation: No exception
    """
    class TraceNet(ms.nn.Cell):
        def __init__(self):
            super(TraceNet, self).__init__()
            self.x = ms.Tensor([5.0, 6.0])

        @ms.jit(capture_mode="trace")
        def construct(self, x, y):
            a = ms.Tensor([7.0, 8.0])
            z = x + a
            z = z.argmax()
            z = z - self.x
            z = z.tanh()
            z = -z
            z = z @ y

            return z

    class JitNet(ms.nn.Cell):
        def __init__(self):
            super(JitNet, self).__init__()
            self.x = ms.Tensor([5.0, 6.0])

        @jit
        def construct(self, x, y):
            a = ms.Tensor([7.0, 8.0])
            z = x + a
            z = z.argmax()
            z = z - self.x
            z = z.tanh()
            z = -z
            z = z @ y
            return z

    class GradNet(ms.nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, x, y):
            z1 = x * y
            z2 = x + y
            z3 = self.net(z1, z2)
            return z3 * z3

    trace_net = TraceNet()
    jit_net = JitNet()
    trace_grad_net = GradNet(trace_net)
    jit_grad_net = GradNet(jit_net)
    res1 = trace_net(ms.Tensor([1.0, 2.0]), ms.Tensor([3.0, 4.0]))
    res2 = jit_net(ms.Tensor([1.0, 2.0]), ms.Tensor([3.0, 4.0]))
    res3 = grad(trace_grad_net)(ms.Tensor([1.0, 2.0]), ms.Tensor([3.0, 4.0]))
    res4 = grad(jit_grad_net)(ms.Tensor([1.0, 2.0]), ms.Tensor([3.0, 4.0]))
    print(f'res1: {res1}, res2: {res2}, res3: {res3}, res4: {res4}')
    assert np.allclose(res1.asnumpy(), res2.asnumpy())
    assert np.allclose(res3.asnumpy(), res4.asnumpy())


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
def test_trace_8():
    """
    Feature: JIT trace function
    Description: JIT trace function with pyboost ops
    Expectation: No exception
    """
    class TraceNet(ms.nn.Cell):
        def __init__(self):
            super(TraceNet, self).__init__()
            self.x = ms.Tensor([5.0, 6.0])

        @ms.jit(capture_mode="trace")
        def construct(self, x, y):
            a = ms.Tensor([7.0, 8.0])
            z = x + a
            z = z.argmax()
            z = z - self.x
            z = z.tanh()
            z = -z
            z = z @ y
            return z * y

    class JitNet(ms.nn.Cell):
        def __init__(self):
            super(JitNet, self).__init__()
            self.x = ms.Tensor([5.0, 6.0])

        @jit
        def construct(self, x, y):
            a = ms.Tensor([7.0, 8.0])
            z = x + a
            z = z.argmax()
            z = z - self.x
            z = z.tanh()
            z = -z
            z = z @ y
            return z * y

    class GradNet(ms.nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, x, y):
            z1 = x * y
            z2 = x + y
            z3 = self.net(z1, z2)
            return z3 * z3

    trace_net = TraceNet()
    jit_net = JitNet()
    trace_grad_net = GradNet(trace_net)
    jit_grad_net = GradNet(jit_net)
    res1 = trace_net(ms.Tensor([1.0, 2.0]), ms.Tensor([3.0, 4.0]))
    res2 = jit_net(ms.Tensor([1.0, 2.0]), ms.Tensor([3.0, 4.0]))
    res3 = grad(trace_grad_net)(ms.Tensor([1.0, 2.0]), ms.Tensor([3.0, 4.0]))
    res4 = grad(jit_grad_net)(ms.Tensor([1.0, 2.0]), ms.Tensor([3.0, 4.0]))
    print(f'res1: {res1}, res2: {res2}, res3: {res3}, res4: {res4}')
    assert np.allclose(res1.asnumpy(), res2.asnumpy())
    assert np.allclose(res3.asnumpy(), res4.asnumpy())


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_trace_9():
    """
    Feature: JIT trace function
    Description: JIT trace function with pyboost ops
    Expectation: No exception
    """
    class TraceNet(ms.nn.Cell):
        def __init__(self):
            super(TraceNet, self).__init__()
            self.x = ms.Tensor([5.0, 6.0])

        def construct(self, x, y):
            a = ms.Tensor([7.0, 8.0])
            z = x + a
            z = z.argmax()
            z = z - self.x
            z = z.tanh()
            z = -z
            z = z @ y
            return z * y

    class JitNet(ms.nn.Cell):
        def __init__(self):
            super(JitNet, self).__init__()
            self.x = ms.Tensor([5.0, 6.0])

        @jit
        def construct(self, x, y):
            a = ms.Tensor([7.0, 8.0])
            z = x + a
            z = z.argmax()
            z = z - self.x
            z = z.tanh()
            z = -z
            z = z @ y
            return z * y

    class GradNet(ms.nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        @ms.jit(capture_mode="trace")
        def construct(self, x, y):
            z1 = x * y
            z2 = x + y
            z3 = self.net(z1, z2)
            return z3 * z3

    trace_net = TraceNet()
    jit_net = JitNet()
    trace_grad_net = GradNet(trace_net)
    jit_grad_net = GradNet(jit_net)
    res1 = trace_net(ms.Tensor([1.0, 2.0]), ms.Tensor([3.0, 4.0]))
    res2 = jit_net(ms.Tensor([1.0, 2.0]), ms.Tensor([3.0, 4.0]))
    res3 = grad(trace_grad_net)(ms.Tensor([1.0, 2.0]), ms.Tensor([3.0, 4.0]))
    res4 = grad(jit_grad_net)(ms.Tensor([1.0, 2.0]), ms.Tensor([3.0, 4.0]))
    print(f'res1: {res1}, res2: {res2}, res3: {res3}, res4: {res4}')
    assert np.allclose(res1.asnumpy(), res2.asnumpy())
    assert np.allclose(res3.asnumpy(), res4.asnumpy())


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='essential')
def test_trace_10():
    """
    Feature: JIT trace function
    Description: JIT trace function with pyboost ops
    Expectation: No exception
    """
    class TraceNet(ms.nn.Cell):
        def __init__(self):
            super(TraceNet, self).__init__()
            self.x = ms.Tensor([5.0, 6.0])

        @ms.jit(capture_mode="trace")
        def construct(self, x, y):
            a = ms.Tensor([7.0, 8.0])
            z = x + a
            z = z.argmax()
            z = z - self.x
            z = z.tanh()
            z = -z
            z = z @ y
            return z * y

    class JitNet(ms.nn.Cell):
        def __init__(self):
            super(JitNet, self).__init__()
            self.x = ms.Tensor([5.0, 6.0])

        def construct(self, x, y):
            a = ms.Tensor([7.0, 8.0])
            z = x + a
            z = z.argmax()
            z = z - self.x
            z = z.tanh()
            z = -z
            z = z @ y
            return z * y

    class GradNet(ms.nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        @ms.jit(capture_mode="ast")
        def construct(self, x, y):
            z1 = x * y
            z2 = x + y
            z3 = self.net(z1, z2)
            return z3 * z3

    trace_net = TraceNet()
    jit_net = JitNet()
    trace_grad_net = GradNet(trace_net)
    jit_grad_net = GradNet(jit_net)
    res1 = trace_net(ms.Tensor([1.0, 2.0]), ms.Tensor([3.0, 4.0]))
    res2 = jit_net(ms.Tensor([1.0, 2.0]), ms.Tensor([3.0, 4.0]))
    res3 = grad(trace_grad_net)(ms.Tensor([1.0, 2.0]), ms.Tensor([3.0, 4.0]))
    res4 = grad(jit_grad_net)(ms.Tensor([1.0, 2.0]), ms.Tensor([3.0, 4.0]))
    print(f'res1: {res1}, res2: {res2}, res3: {res3}, res4: {res4}')
    assert np.allclose(res1.asnumpy(), res2.asnumpy())
    assert np.allclose(res3.asnumpy(), res4.asnumpy())


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential')
def test_trace_11():
    """
    Feature: JIT trace function
    Description: JIT trace function
    Expectation: No exception
    """

    class ParamNetMultipleOutputs(nn.Cell):
        def __init__(self):
            super(ParamNetMultipleOutputs, self).__init__()
            self.w1 = Parameter(Tensor([2., 2.], mstype.float32), name="w1")
            self.w2 = Parameter(Tensor([3., 3.], mstype.float32), name="w2")

        def construct(self, x):
            res = x * self.w1 * self.w2
            return res, x, self.w1
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(np.array([1, 2]).astype(np.float32))
    net = ParamNetMultipleOutputs()
    weights = ParameterTuple(net.trainable_params())
    expect_grad_input = np.array([6, 6]).astype(np.float32)
    expect_grad_weight1 = np.array([3, 6]).astype(np.float32)
    expect_grad_weight2 = np.array([2, 4]).astype(np.float32)
    expect_value0 = np.array([6, 12]).astype(np.float32)
    expect_value1 = np.array([1, 2]).astype(np.float32)
    expect_value2 = np.array([2, 2]).astype(np.float32)

    @ms.jit(capture_mode="trace")
    def trace_func(x):
        return ops.value_and_grad(net, 0, weights, True)(x)
    value, gradient = trace_func(x)
    assert np.allclose(value[0].asnumpy(), expect_value0)
    assert np.allclose(value[1].asnumpy(), expect_value1)
    assert np.allclose(value[2].asnumpy(), expect_value2)
    assert np.allclose(gradient[0].asnumpy(), expect_grad_input)
    assert np.allclose(gradient[1][0].asnumpy(), expect_grad_weight1)
    assert np.allclose(gradient[1][1].asnumpy(), expect_grad_weight2)
