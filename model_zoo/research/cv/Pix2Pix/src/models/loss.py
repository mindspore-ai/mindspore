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
# ===========================================================================

"""
    Define losses.
"""

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import functional as F
import mindspore.ops.operations as P
from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean, _get_parallel_mode)
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.nn.loss.loss import _Loss
from src.utils.config import get_args

args = get_args()

class SigmoidCrossEntropyWithLogits(_Loss):
    def __init__(self):
        super(SigmoidCrossEntropyWithLogits, self).__init__()
        self.cross_entropy = P.SigmoidCrossEntropyWithLogits()

    def construct(self, data, label):
        x = self.cross_entropy(data, label)
        return self.get_loss(x)

class D_Loss(_Loss):
    """
        Define Dloss.
    """
    def __init__(self, reduction="mean"):
        super(D_Loss, self).__init__(reduction)
        self.sig = SigmoidCrossEntropyWithLogits()
        self.ones = ops.OnesLike()
        self.zeros = ops.ZerosLike()
        self.LAMBDA_Dis = args.LAMBDA_Dis

    def construct(self, pred1, pred0):
        loss = self.sig(pred1, self.ones(pred1)) + self.sig(pred0, self.zeros(pred0))
        dis_loss = loss * self.LAMBDA_Dis
        return dis_loss

class D_WithLossCell(nn.Cell):
    """
        Define D_WithLossCell.
    """
    def __init__(self, backbone, loss_fn):
        super(D_WithLossCell, self).__init__(auto_prefix=True)
        self.netD = backbone.netD
        self.netG = backbone.netG
        self._loss_fn = loss_fn

    def construct(self, realA, realB):
        fakeB = self.netG(realA)
        pred1 = self.netD(realA, realB)
        pred0 = self.netD(realA, fakeB)
        return self._loss_fn(pred1, pred0)

class G_Loss(_Loss):
    """
        Define Gloss.
    """
    def __init__(self, reduction="mean"):
        super(G_Loss, self).__init__(reduction)
        self.sig = SigmoidCrossEntropyWithLogits()
        self.l1_loss = nn.L1Loss()
        self.ones = ops.OnesLike()
        self.LAMBDA_GAN = args.LAMBDA_GAN
        self.LAMBDA_L1 = args.LAMBDA_L1

    def construct(self, fakeB, realB, pred0):
        loss_1 = self.sig(pred0, self.ones(pred0))
        loss_2 = self.l1_loss(fakeB, realB)
        loss = loss_1 * self.LAMBDA_GAN + loss_2 * self.LAMBDA_L1
        return loss

class G_WithLossCell(nn.Cell):
    """
        Define G_WithLossCell.
    """
    def __init__(self, backbone, loss_fn):
        super(G_WithLossCell, self).__init__(auto_prefix=True)
        self.netD = backbone.netD
        self.netG = backbone.netG
        self._loss_fn = loss_fn

    def construct(self, realA, realB):
        fakeB = self.netG(realA)
        pred0 = self.netD(realA, fakeB)
        return self._loss_fn(fakeB, realB, pred0)

class TrainOneStepCell(nn.Cell):
    """
        Define TrainOneStepCell.
    """
    def __init__(self, loss_netD, loss_netG, optimizerD, optimizerG, sens=1, auto_prefix=True):
        super(TrainOneStepCell, self).__init__(auto_prefix=auto_prefix)
        self.loss_netD = loss_netD        # loss network
        self.loss_netD.set_grad()
        self.loss_netD.add_flags(defer_inline=True)

        self.loss_netG = loss_netG
        self.loss_netG.set_grad()
        self.loss_netG.add_flags(defer_inline=True)

        self.weights_G = optimizerG.parameters
        self.optimizerG = optimizerG
        self.weights_D = optimizerD.parameters
        self.optimizerD = optimizerD

        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

        # Parallel processing
        self.reducer_flag = False
        self.grad_reducer_G = F.identity
        self.grad_reducer_D = F.identity
        self.parallel_mode = _get_parallel_mode()
        if self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer_G = DistributedGradReducer(self.weights_G, mean, degree)
            self.grad_reducer_D = DistributedGradReducer(self.weights_D, mean, degree)

    def set_sens(self, value):
        self.sens = value

    def construct(self, realA, realB):
        """
            Define TrainOneStepCell.
        """
        d_loss = self.loss_netD(realA, realB)
        g_loss = self.loss_netG(realA, realB)

        d_sens = ops.Fill()(ops.DType()(d_loss), ops.Shape()(d_loss), self.sens)
        d_grads = self.grad(self.loss_netD, self.weights_D)(realA, realB, d_sens)
        d_res = ops.depend(d_loss, self.optimizerD(d_grads))

        g_sens = ops.Fill()(ops.DType()(g_loss), ops.Shape()(g_loss), self.sens)
        g_grads = self.grad(self.loss_netG, self.weights_G)(realA, realB, g_sens)
        g_res = ops.depend(g_loss, self.optimizerG(g_grads))
        return d_res, g_res
