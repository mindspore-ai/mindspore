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
"""cell define"""
from mindspore import nn
import mindspore.ops.operations as P
import mindspore.ops.functional as F
import mindspore.ops.composite as C
from mindspore.parallel._utils import _get_device_num, _get_parallel_mode, _get_gradients_mean
from mindspore.context import ParallelMode
from mindspore.ops import OnesLike, ZerosLike
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer


class GenWithLossCell(nn.Cell):
    """GenWithLossCell"""
    def __init__(self, netG, netD, auto_prefix=True):
        super(GenWithLossCell, self).__init__(auto_prefix=auto_prefix)
        self.netG = netG
        self.netD = netD
        self.loss_fn = nn.BCELoss(reduction="mean")


    def construct(self, latent_code, label):
        """cgan construct"""
        fake_data = self.netG(latent_code, label)

        # loss
        fake_out = self.netD(fake_data, label)
        ones = OnesLike()(fake_out)
        loss_G = self.loss_fn(fake_out, ones)

        return loss_G


class DisWithLossCell(nn.Cell):
    """DisWithLossCell"""
    def __init__(self, netG, netD, auto_prefix=True):
        super(DisWithLossCell, self).__init__(auto_prefix=auto_prefix)
        self.netG = netG
        self.netD = netD
        self.loss_fn = nn.BCELoss(reduction="mean")

    def construct(self, real_data, latent_code, label):
        """construct"""
        # fake_data
        fake_data = self.netG(latent_code, label)
        # fake_loss
        fake_out = self.netD(fake_data, label)
        zeros = ZerosLike()(fake_out)
        fake_loss = self.loss_fn(fake_out, zeros)
        # real loss
        real_out = self.netD(real_data, label)
        ones = OnesLike()(real_out)
        real_loss = self.loss_fn(real_out, ones)

        # d loss
        loss_D = real_loss + fake_loss

        return loss_D


class TrainOneStepCell(nn.Cell):
    """define TrainOneStepCell"""
    def __init__(self,
                 netG,
                 netD,
                 optimizerG: nn.Optimizer,
                 optimizerD: nn.Optimizer,
                 sens=1.0,
                 auto_prefix=True):

        super(TrainOneStepCell, self).__init__(auto_prefix=auto_prefix)
        self.netG = netG
        self.netG.set_grad()
        self.netG.add_flags(defer_inline=True)

        self.netD = netD
        self.netD.set_grad()
        self.netD.add_flags(defer_inline=True)

        self.weights_G = optimizerG.parameters
        self.optimizerG = optimizerG
        self.weights_D = optimizerD.parameters
        self.optimizerD = optimizerD

        self.grad = C.GradOperation(get_by_list=True, sens_param=True)

        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer_G = F.identity
        self.grad_reducer_D = F.identity
        self.parallel_mode = _get_parallel_mode()
        if self.parallel_mode in (ParallelMode.DATA_PARALLEL,
                                  ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer_G = DistributedGradReducer(
                self.weights_G, mean, degree)
            self.grad_reducer_D = DistributedGradReducer(
                self.weights_D, mean, degree)

    def trainD(self, real_data, latent_code, label, loss):
        """trainD"""
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.netD, self.weights_D)(real_data, latent_code, label, sens)
        grads = self.grad_reducer_D(grads)
        return F.depend(loss, self.optimizerD(grads))

    def trainG(self, latent_code, label, loss):
        """trainG"""
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.netG, self.weights_G)(latent_code, label, sens)
        grads = self.grad_reducer_G(grads)
        return F.depend(loss, self.optimizerG(grads))

    def construct(self, real_data, latent_code, label):
        """construct"""
        loss_D = self.netD(real_data, latent_code, label)
        loss_G = self.netG(latent_code, label)
        d_out = self.trainD(real_data, latent_code, label, loss_D)
        g_out = self.trainG(latent_code, label, loss_G)
        return d_out, g_out
