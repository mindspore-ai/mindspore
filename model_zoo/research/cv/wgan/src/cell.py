# Copyright 2021 Huawei Technologies Co., Ltd
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

""" Train one step """
import mindspore.nn as nn
import mindspore.ops.composite as C
import mindspore.ops.operations as P
import mindspore.ops.functional as F
from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean,
                                       _get_parallel_mode)
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer


class GenWithLossCell(nn.Cell):
    """Generator with loss(wrapped)"""
    def __init__(self, netG, netD):
        super(GenWithLossCell, self).__init__()
        self.netG = netG
        self.netD = netD

    def construct(self, noise):
        """construct"""
        fake = self.netG(noise)
        errG = self.netD(fake)
        loss_G = errG
        return loss_G


class DisWithLossCell(nn.Cell):
    """ Discriminator with loss(wrapped) """
    def __init__(self, netG, netD):
        super(DisWithLossCell, self).__init__()
        self.netG = netG
        self.netD = netD

    def construct(self, real, noise):
        """construct"""
        errD_real = self.netD(real)
        fake = self.netG(noise)
        errD_fake = self.netD(fake)
        loss_D = errD_real - errD_fake
        return loss_D


class ClipParameter(nn.Cell):
    """ Clip the parameter """
    def __init__(self):
        super(ClipParameter, self).__init__()
        self.cast = P.Cast()
        self.dtype = P.DType()

    def construct(self, params, clip_lower, clip_upper):
        """construct"""
        new_params = ()
        for param in params:
            dt = self.dtype(param)
            t = C.clip_by_value(param, self.cast(F.tuple_to_array((clip_lower,)), dt),
                                self.cast(F.tuple_to_array((clip_upper,)), dt))
            new_params = new_params + (t,)

        return new_params


class GenTrainOneStepCell(nn.Cell):
    """ Generator TrainOneStepCell """
    def __init__(self, netG, netD,
                 optimizerG: nn.Optimizer,
                 sens=1.0):
        super(GenTrainOneStepCell, self).__init__()
        self.netD = netD
        self.netD.set_train(False)
        self.netD.set_grad(False)
        self.weights_G = optimizerG.parameters
        self.optimizerG = optimizerG
        self.net = GenWithLossCell(netG, netD)
        self.net.set_train()
        self.net.set_grad()

        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

        # parallel process
        self.reducer_flag = False
        self.grad_reducer_G = F.identity
        self.parallel_mode = _get_parallel_mode()
        if self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer_G = DistributedGradReducer(self.weights_G, mean, degree)  # A distributed optimizer.

    def construct(self, noise):
        """ construct """
        loss_G = self.net(noise)
        sens = P.Fill()(P.DType()(loss_G), P.Shape()(loss_G), self.sens)
        grads = self.grad(self.net, self.weights_G)(noise, sens)
        if self.reducer_flag:
            grads = self.grad_reducer_G(grads)
        return F.depend(loss_G, self.optimizerG(grads))


_my_adam_opt = C.MultitypeFuncGraph("_my_adam_opt")


@_my_adam_opt.register("Tensor", "Tensor")
def _update_run_op(param, param_clipped):
    param_clipped = F.depend(param_clipped, F.assign(param, param_clipped))
    return param_clipped


class DisTrainOneStepCell(nn.Cell):
    """ Discriminator TrainOneStepCell """
    def __init__(self, netG, netD,
                 optimizerD: nn.Optimizer,
                 clip_lower=-0.01, clip_upper=0.01, sens=1.0):
        super(DisTrainOneStepCell, self).__init__()
        self.weights_D = optimizerD.parameters
        self.clip_parameters = ClipParameter()
        self.optimizerD = optimizerD
        self.net = DisWithLossCell(netG, netD)
        self.net.set_train()
        self.net.set_grad()

        self.reduce_flag = False
        self.op_cast = P.Cast()
        self.hyper_map = C.HyperMap()

        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper

        # parallel process
        self.reducer_flag = False
        self.grad_reducer_D = F.identity
        self.parallel_mode = _get_parallel_mode()
        if self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer_D = DistributedGradReducer(self.weights_D, mean, degree)  # A distributed optimizer.

    def construct(self, real, noise):
        """ construct """
        loss_D = self.net(real, noise)
        sens = P.Fill()(P.DType()(loss_D), P.Shape()(loss_D), self.sens)
        grads = self.grad(self.net, self.weights_D)(real, noise, sens)
        if self.reducer_flag:
            grads = self.grad_reducer_D(grads)

        upd = self.optimizerD(grads)
        weights_D_cliped = self.clip_parameters(self.weights_D, self.clip_lower, self.clip_upper)
        res = self.hyper_map(F.partial(_my_adam_opt), self.weights_D, weights_D_cliped)
        res = F.depend(upd, res)
        return F.depend(loss_D, res)
