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
"""Define loss function for StarGAN"""
import numpy as np

import mindspore.ops as ops
import mindspore.ops.operations as P

from mindspore import nn, Tensor
from mindspore import dtype as mstype
from mindspore.ops import constexpr


@constexpr
def generate_tensor(batch_size):
    np_array = np.random.randn(batch_size, 1, 1, 1)
    return Tensor(np_array, mstype.float32)


class ClassificationLoss(nn.Cell):
    """Define classification loss for StarGAN"""
    def __init__(self, dataset='CelebA'):
        super().__init__()
        self.BCELoss = P.BinaryCrossEntropy(reduction='sum')
        self.cross_entropy = P.SoftmaxCrossEntropyWithLogits()
        self.dataset = dataset
        self.bec = nn.BCELoss(reduction='sum')

    def construct(self, pred, label):
        if self.dataset == 'CelebA':
            weight = ops.Ones()(pred.shape, mstype.float32)
            pred_ = P.Sigmoid()(pred)
            x = self.BCELoss(pred_, label, weight) / pred.shape[0]

        else:
            x = self.cross_entropy(pred, label)
        return x


class GradientWithInput(nn.Cell):
    """Get Discriminator Gradient with Input"""
    def __init__(self, discrimator):
        super(GradientWithInput, self).__init__()
        self.reduce_sum = ops.ReduceSum()
        self.discrimator = discrimator

    def construct(self, interpolates):
        decision_interpolate, _ = self.discrimator(interpolates)
        decision_interpolate = self.reduce_sum(decision_interpolate, 0)
        return decision_interpolate


class WGANGPGradientPenalty(nn.Cell):
    """Define WGAN loss for StarGAN"""
    def __init__(self, discrimator):
        super(WGANGPGradientPenalty, self).__init__()
        self.gradient_op = ops.GradOperation()

        self.reduce_sum = ops.ReduceSum()
        self.reduce_sum_keep_dim = ops.ReduceSum(keep_dims=True)
        self.sqrt = ops.Sqrt()
        self.discrimator = discrimator
        self.gradientWithInput = GradientWithInput(discrimator)

    def construct(self, x_real, x_fake):
        """get gradient penalty"""
        batch_size = x_real.shape[0]
        alpha = generate_tensor(batch_size)
        alpha = alpha.expand_as(x_real)
        x_fake = ops.functional.stop_gradient(x_fake)
        x_hat = (alpha * x_real + (1 - alpha) * x_fake)

        gradient = self.gradient_op(self.gradientWithInput)(x_hat)
        gradient_1 = ops.reshape(gradient, (batch_size, -1))
        gradient_1 = self.sqrt(self.reduce_sum(gradient_1*gradient_1, 1))
        gradient_penalty = self.reduce_sum((gradient_1 - 1.0)**2) / x_real.shape[0]
        return gradient_penalty


class GeneratorLoss(nn.Cell):
    """Define total Generator loss"""
    def __init__(self, args, generator, discriminator):
        super(GeneratorLoss, self).__init__()
        self.net_G = generator
        self.net_D = discriminator
        self.cyc_loss = P.ReduceMean()
        self.rec_loss = nn.L1Loss("mean")
        self.cls_loss = ClassificationLoss()

        self.lambda_rec = args.lambda_rec
        self.lambda_cls = args.lambda_cls

    def construct(self, x_real, c_org, c_trg):
        """Get generator loss"""
        # Original to Target
        x_fake = self.net_G(x_real, c_trg)
        fake_src, fake_cls = self.net_D(x_fake)

        G_fake_loss = - self.cyc_loss(fake_src)
        G_fake_cls_loss = self.cls_loss(fake_cls, c_trg)

        # Target to Original
        x_rec = self.net_G(x_fake, c_org)
        G_rec_loss = self.rec_loss(x_real, x_rec)

        g_loss = G_fake_loss + self.lambda_cls * G_fake_cls_loss + self.lambda_rec * G_rec_loss

        return (x_fake, g_loss, G_fake_loss, G_fake_cls_loss, G_rec_loss)


class DiscriminatorLoss(nn.Cell):
    """Define total discriminator loss"""
    def __init__(self, args, generator, discriminator):
        super(DiscriminatorLoss, self).__init__()
        self.net_G = generator
        self.net_D = discriminator
        self.cyc_loss = P.ReduceMean()
        self.cls_loss = ClassificationLoss()
        self.WGANLoss = WGANGPGradientPenalty(discriminator)

        self.lambda_rec = Tensor(args.lambda_rec)
        self.lambda_cls = Tensor(args.lambda_cls)
        self.lambda_gp = Tensor(args.lambda_gp)

    def construct(self, x_real, c_org, c_trg):
        """Get discriminator loss"""
        # Compute loss with real images
        real_src, real_cls = self.net_D(x_real)

        D_real_loss = - self.cyc_loss(real_src)
        D_real_cls_loss = self.cls_loss(real_cls, c_org)

        # Compute loss with fake images
        x_fake = self.net_G(x_real, c_trg)
        fake_src, _ = self.net_D(x_fake)
        D_fake_loss = self.cyc_loss(fake_src)

        D_gp_loss = self.WGANLoss(x_real, x_fake)

        d_loss = D_real_loss + D_fake_loss + self.lambda_cls * D_real_cls_loss + self.lambda_gp *D_gp_loss

        return (d_loss, D_real_loss, D_fake_loss, D_real_cls_loss, D_gp_loss)
