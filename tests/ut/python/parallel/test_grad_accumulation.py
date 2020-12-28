# Copyright 2020 Huawei Technologies Co., Ltd
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
import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import context, Tensor, Parameter
from mindspore.nn import Cell, Momentum, Norm
from mindspore.train import Model
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.context import ParallelMode

from tests.dataset_mock import MindData


class Dataset(MindData):
    def __init__(self, predict, label, length=3):
        super(Dataset, self).__init__(size=length)
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


get_square_sum = C.MultitypeFuncGraph("get_square_sum")
@get_square_sum.register("Tensor")
def _get_square_sum(grad):
    norm = P.ReduceSum(False)(F.square(grad), ())
    norm = F.expand_dims(F.cast(norm, mstype.float32), 0)
    return norm


apply_global_norm = C.MultitypeFuncGraph("apply_global_norm")
@apply_global_norm.register("Tensor", "Tensor", "Tensor")
def _apply_global_norm(clip_norm, global_norm, grad):
    grad = grad * clip_norm / global_norm
    return grad


class GlobalNorm(Cell):
    """
    Calculate the global norm value of given tensors
    """
    def __init__(self):
        super(GlobalNorm, self).__init__()
        self.norm = Norm()
        self.hyper_map = C.HyperMap()

    def construct(self, grads):
        square_sum = self.hyper_map(get_square_sum, grads)
        global_norms = F.sqrt(F.addn(square_sum) / F.scalar_to_array(len(square_sum)))
        return global_norms


class ClipByGlobalNorm(Cell):
    """
    Clip grads by global norm
    """
    def __init__(self, clip_norm=1.0):
        super(ClipByGlobalNorm, self).__init__()
        self.global_norm = GlobalNorm()
        self.clip_norm = Tensor([clip_norm], mstype.float32)
        self.hyper_map = C.HyperMap()

    def construct(self, grads):
        global_norm = self.global_norm(grads)
        cond = P.GreaterEqual()(global_norm, self.clip_norm)
        global_norm = F.select(cond, global_norm, self.clip_norm)
        grads = self.hyper_map(F.partial(apply_global_norm, self.clip_norm, global_norm), grads)
        return grads


cast = P.Cast()
update_accu_grads = C.MultitypeFuncGraph("update_accu_grads")


@update_accu_grads.register("Tensor", "Tensor")
def _update_accu_grads(accu_grad, grad):
    succ = True
    return F.depend(succ, F.assign_add(accu_grad, cast(grad, mstype.float32)))


zeroslike = P.ZerosLike()
reset_accu_grads = C.MultitypeFuncGraph("reset_accu_grads")


@reset_accu_grads.register("Tensor")
def _reset_accu_grads(accu_grad):
    succ = True
    return F.depend(succ, F.assign(accu_grad, zeroslike(accu_grad)))


grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)


class TrainAccumulateStepsWithLossScaleCell(Cell):
    """
    Encapsulation class of bert network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph. To mimic higher batch size, gradients are
    accumulated N times before weight update.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
        accumulation_steps (int): Number of accumulation steps before gradient update. The global batch size =
                                batch_size * accumulation_steps. Default: 1.
    """
    def __init__(self, network, optimizer, scale_update_cell=None, accumulation_steps=4):
        super(TrainAccumulateStepsWithLossScaleCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.one = Tensor(np.array([1]).astype(np.int32))
        self.zero = Tensor(np.array([0]).astype(np.int32))
        self.local_step = Parameter(initializer(0, [1], mstype.int32), name="local_step")
        self.accu_grads = self.weights.clone(prefix="accu_grads", init='zeros')
        self.accu_overflow = Parameter(initializer(0, [1], mstype.int32))
        self.accu_loss = Parameter(initializer(0, [1], mstype.float32))

        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.reducer_flag = False
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = F.identity
        self.degree = 1
        if self.reducer_flag:
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, False, self.degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.overflow_reducer = F.identity
        if self.is_distributed:
            self.overflow_reducer = P.AllReduce()
        self.cast = P.Cast()
        self.alloc_status = P.NPUAllocFloatStatus()
        self.get_status = P.NPUGetFloatStatus()
        self.clear_before_grad = P.NPUClearFloatStatus()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.logical_or = P.LogicalOr()
        self.not_equal = P.NotEqual()
        self.select = P.Select()
        self.reshape = P.Reshape()
        self.hyper_map = C.HyperMap()
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32))

    @C.add_flags(has_effect=True)
    def construct(self, x, b, sens=None):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(x, b)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens

        # update accumulation parameters
        is_accu_step = self.not_equal(self.local_step, self.accumulation_steps)
        self.local_step = self.select(is_accu_step, self.local_step + self.one, self.one)
        self.accu_loss = self.select(is_accu_step, self.accu_loss + loss, loss)
        mean_loss = self.accu_loss / self.local_step
        is_accu_step = self.not_equal(self.local_step, self.accumulation_steps)

        # alloc status and clear should be right before gradoperation
        init = self.alloc_status()
        self.clear_before_grad(init)
        grads = self.grad(self.network, weights)(x, b, self.cast(scaling_sens, mstype.float32))

        accu_succ = self.hyper_map(update_accu_grads, self.accu_grads, grads)
        mean_loss = F.depend(mean_loss, accu_succ)

        self.get_status(init)
        flag_sum = self.reduce_sum(init, (0,))
        overflow = self.less_equal(self.base, flag_sum)
        overflow = self.logical_or(self.not_equal(self.accu_overflow, self.zero), overflow)
        accu_overflow = self.select(overflow, self.one, self.zero)
        self.accu_overflow = self.select(is_accu_step, accu_overflow, self.zero)
        is_accu_step = self.reshape(is_accu_step, (()))

        if is_accu_step:
            succ = False
        else:
            # apply grad reducer on grads
            grads = self.grad_reducer(self.accu_grads)
            scaling = scaling_sens * self.degree * self.accumulation_steps
            grads = self.hyper_map(F.partial(grad_scale, scaling), grads)
            grads = ClipByGlobalNorm()(grads)
            accu_overflow = self.overflow_reducer(accu_overflow)
            F.control_depend(grads, accu_overflow)
            overflow = self.less_equal(self.base, accu_overflow)
            accu_succ = self.hyper_map(reset_accu_grads, self.accu_grads)
            overflow = F.depend(overflow, accu_succ)
            overflow = self.reshape(overflow, (()))
            if sens is None:
                overflow = self.loss_scaling_manager(self.loss_scale, overflow)
            if overflow:
                succ = False
            else:
                succ = self.optimizer(grads)

        ret = (mean_loss, overflow, scaling_sens)
        return F.depend(ret, succ)


class Net(Cell):
    def __init__(self, weight, strategy=None):
        super().__init__()
        self.mul = P.Mul().shard(strategy)
        self.weight = Parameter(weight, "w1")
        self.relu = P.ReLU()
        self.reduce_sum = P.ReduceSum(keep_dims=True)

    def construct(self, x, b):
        out = self.mul(x, self.weight)
        out = self.relu(out)
        out = self.reduce_sum(out)
        return out


_x = Tensor(np.ones([2]), dtype=ms.float32)
_b = Tensor(np.ones([16]), dtype=ms.float32)
_w1 = Tensor(np.ones([16]), dtype=ms.float32)


def compile_net(net, grad_accumulation_step):
    context.set_context(save_graphs=True)
    learning_rate = 0.1
    momentum = 0.9
    epoch_size = 2
    dataset = Dataset(_x, _b)
    opt = Momentum(net.trainable_params(), learning_rate, momentum)
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=65536, scale_factor=2, scale_window=1000)
    net_wrap = TrainAccumulateStepsWithLossScaleCell(net, opt, scale_update_cell=update_cell,
                                                     accumulation_steps=grad_accumulation_step)
    model = Model(net_wrap)
    model.train(epoch_size, dataset, dataset_sink_mode=False)
    context.reset_auto_parallel_context()


def test_grad_accumulation():
    grad_accumulation_step = 4
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0,
                                      grad_accumulation_step=grad_accumulation_step)
    strategy = ((2,), (2,))
    net = Net(_w1, strategy)
    compile_net(net, grad_accumulation_step)
