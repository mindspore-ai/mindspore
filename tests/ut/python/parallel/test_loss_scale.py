# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
from mindspore import Tensor
from mindspore import context
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
from mindspore.context import ParallelMode
from tests.dataset_mock import MindData

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0
clip_grad = C.MultitypeFuncGraph("clip_grad")
grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)


update_cell = DynamicLossScaleUpdateCell(
    loss_scale_value=65536, scale_factor=2, scale_window=1000)


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = ops.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                     F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
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
        self.grad_reducer = self.identity
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

    def identity(self, x):
        return x

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

    def construct(self, x):
        out = self.exp(x)
        for layer in self.layers:
            layer_out = layer(out)
            out = layer_out
        out = self.mean(out, -1)
        return out


class Net2(nn.Cell):
    def __init__(self):
        super(Net2, self).__init__()
        self.matmul = P.MatMul()
        self.relu = P.ReLU()
        self.matmul_weight = Parameter(
            Tensor(np.ones([64, 64]), dtype=ms.float32), name="weight")

    def construct(self, x, b):
        out = self.matmul(x, self.matmul_weight)
        out = self.relu(out)
        return out


def test_loss_scale():
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(
        parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, device_num=8)
    predict = Tensor(np.ones([64, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64,]), dtype=ms.int32)
    dataset = DatasetLenet(predict, label)
    net = Net()
    opt = Momentum(filter(lambda x: x.requires_grad,
                          net.get_parameters()), 0.01, 0.9)
    net = TrainOneStepWithLossScaleCell(net, opt, update_cell)
    model = Model(network=net)
    model.train(2, dataset, dataset_sink_mode=False)


def test_loss_scale2():
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(
        parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, device_num=8)
    predict = Tensor(np.ones([64, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64,]), dtype=ms.int32)
    dataset = DatasetLenet(predict, label)
    net = Net2()
    opt = Momentum(filter(lambda x: x.requires_grad,
                          net.get_parameters()), 0.01, 0.9)
    net = nn.TrainOneStepWithLossScaleCell(net, opt, update_cell)
    model = Model(network=net)
    model.train(2, dataset, dataset_sink_mode=False)
