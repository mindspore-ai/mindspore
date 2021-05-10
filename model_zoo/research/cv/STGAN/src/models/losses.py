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
""" losses """
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.ops.operations as P

from mindspore import Tensor

from src.utils.tools import is_version_satisfied_1_2_0


class GeneratorLoss(nn.Cell):
    """ GeneratorLoss """
    def __init__(self, netG, netD, args, mode):
        super(GeneratorLoss, self).__init__()
        self.args = args
        self.platform = args.platform
        self.use_stu = args.use_stu
        self.mode = mode
        self.netG = netG
        self.netD = netD
        if self.platform == 'Ascend':
            self.cls_loss = nn.BCEWithLogitsLoss(reduction='sum')
        else:
            self.cls_loss = ClassificationLoss()
        self.cyc_loss = P.ReduceMean()

        self.lambda2 = Tensor(args.lambda2)
        self.lambda3 = Tensor(args.lambda3)
        self.attr_mode = args.attr_mode

    def construct(self, real_x, c_org, c_trg, attr_diff):
        """ construct """
        fake_x = self.netG(real_x, attr_diff)
        rec_x = self.netG(real_x, c_org - c_org)
        loss_fake_G, loss_att_fake = self.netD(fake_x)

        loss_adv_G = -loss_fake_G.mean()
        loss_cls_G = self.cls_loss(loss_att_fake,
                                   c_trg) / loss_att_fake.shape[0]
        loss_rec_G = (real_x - rec_x).abs().mean()

        loss_G = loss_adv_G + self.lambda2 * loss_cls_G + self.lambda3 * loss_rec_G

        return (fake_x, loss_G, loss_fake_G.mean(), loss_cls_G, loss_rec_G,
                loss_adv_G)


class DiscriminatorLoss(nn.Cell):
    """ DiscriminatorLoss """
    def __init__(self, netD, netG, args, mode):
        super(DiscriminatorLoss, self).__init__()
        self.mode = mode
        self.netD = netD
        self.netG = netG
        self.platform = args.platform
        self.gradient_penalty = WGANGPGradientPenalty(netD)
        if self.platform == 'Ascend' and is_version_satisfied_1_2_0(
                args.ms_version):
            self.cls_loss = nn.BCEWithLogitsLoss(reduction='sum')
        else:
            self.cls_loss = ClassificationLoss()
        self.cyc_loss = P.ReduceMean()

        self.lambda_gp = Tensor(args.lambda_gp)
        self.lambda1 = Tensor(args.lambda1)
        self.thres_int = Tensor(args.thres_int, ms.float32)
        self.attr_mode = args.attr_mode

    def construct(self, real_x, c_org, c_trg, attr_diff, alpha):
        """ construct """
        loss_real_D, loss_att_real = self.netD(real_x)
        loss_real_D = -loss_real_D.mean()
        loss_cls_D = self.cls_loss(loss_att_real,
                                   c_org) / loss_att_real.shape[0]

        fake_x = self.netG(real_x, attr_diff)
        loss_fake_D, _ = self.netD(ops.functional.stop_gradient(fake_x))
        loss_fake_D = loss_fake_D.mean()

        x_hat = (alpha * real_x + (1 - alpha) * fake_x)
        loss_gp_D = self.gradient_penalty(x_hat)
        loss_adv_D = loss_real_D + loss_fake_D + self.lambda_gp * loss_gp_D
        loss_D = self.lambda1 * loss_cls_D + loss_adv_D

        return (loss_D, loss_real_D, loss_fake_D, loss_cls_D, loss_gp_D,
                loss_adv_D, attr_diff)


class WGANGPGradientPenalty(nn.Cell):
    """ WGANGPGradientPenalty """
    def __init__(self, discriminator):
        super(WGANGPGradientPenalty, self).__init__()
        self.gradient_op = ops.GradOperation()
        self.reduce_sum = ops.ReduceSum()
        self.reduce_sum_keep_dim = ops.ReduceSum(keep_dims=True)
        self.sqrt = ops.Sqrt()
        self.discriminator = discriminator
        self.gradientWithInput = GradientWithInput(discriminator)

    def construct(self, x_hat):
        gradient = self.gradient_op(self.gradientWithInput)(x_hat)
        gradient_1 = ops.reshape(gradient, (x_hat.shape[0], -1))
        gradient_1 = self.sqrt(self.reduce_sum(gradient_1**2, 1))
        gradient_penalty = ((gradient_1 - 1.0)**2).mean()
        return gradient_penalty


class GradientWithInput(nn.Cell):
    def __init__(self, discriminator):
        super(GradientWithInput, self).__init__()
        self.reduce_sum = ops.ReduceSum()
        self.discriminator = discriminator

    def construct(self, interpolates):
        decision_interpolate, _ = self.discriminator(interpolates)
        decision_interpolate = self.reduce_sum(decision_interpolate, 0)
        return decision_interpolate


class ClassificationLoss(nn.Cell):
    def __init__(self):
        super(ClassificationLoss, self).__init__()
        self.BCELoss = nn.BCELoss(reduction='sum')
        self.sigmoid = nn.Sigmoid()

    def construct(self, logit, target):
        logit = self.sigmoid(logit)
        return self.BCELoss(logit, target)
