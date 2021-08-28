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
# ============================================================================s
"""Loss Computation of Generator and Discriminator"""

import numpy as np

import mindspore
import mindspore.ops.operations as P
from mindspore import dtype as mstype
from mindspore import nn, Tensor, ops
from mindspore.ops import constexpr


class ClassificationLoss(nn.Cell):
    """Define classification loss for AttGAN"""

    def __init__(self):
        super().__init__()
        self.bce_loss = P.BinaryCrossEntropy(reduction='sum')

    def construct(self, pred, label):
        weight = ops.Ones()(pred.shape, mindspore.float32)
        pred_ = P.Sigmoid()(pred)
        x = self.bce_loss(pred_, label, weight) / pred.shape[0]
        return x


@constexpr
def generate_tensor(batch_size):
    np_array = np.random.randn(batch_size, 1, 1, 1)
    return Tensor(np_array, mindspore.float32)


class GradientWithInput(nn.Cell):
    """Get Discriminator Gradient with Input"""

    def __init__(self, discriminator):
        super().__init__()
        self.reduce_sum = ops.ReduceSum()
        self.discriminator = discriminator
        self.discriminator.set_train(mode=True)

    def construct(self, interpolates):
        decision_interpolate, _ = self.discriminator(interpolates)
        decision_interpolate = self.reduce_sum(decision_interpolate, 0)
        return decision_interpolate


class WGANGPGradientPenalty(nn.Cell):
    """Define WGAN loss for AttGAN"""

    def __init__(self, discriminator):
        super().__init__()
        self.gradient_op = ops.GradOperation()

        self.reduce_sum = ops.ReduceSum()
        self.reduce_sum_keep_dim = ops.ReduceSum(keep_dims=True)

        self.sqrt = ops.Sqrt()
        self.discriminator = discriminator
        self.GradientWithInput = GradientWithInput(discriminator)

    def construct(self, x_real, x_fake):
        """get gradient penalty"""
        batch_size = x_real.shape[0]
        alpha = generate_tensor(batch_size)
        alpha = alpha.expand_as(x_real)
        x_fake = ops.functional.stop_gradient(x_fake)
        x_hat = x_real + alpha * (x_fake - x_real)

        gradient = self.gradient_op(self.GradientWithInput)(x_hat)
        gradient_1 = ops.reshape(gradient, (batch_size, -1))
        gradient_1 = self.sqrt(self.reduce_sum(gradient_1 * gradient_1, 1))
        gradient_penalty = self.reduce_sum((gradient_1 - 1.0) ** 2) / x_real.shape[0]
        return gradient_penalty


class GenLoss(nn.Cell):
    """Define total Generator loss"""

    def __init__(self, args, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

        self.lambda_1 = Tensor(args.lambda_1, mstype.float32)
        self.lambda_2 = Tensor(args.lambda_2, mstype.float32)
        self.lambda_3 = Tensor(args.lambda_3, mstype.float32)
        self.lambda_gp = Tensor(args.lambda_gp, mstype.float32)

        self.cyc_loss = P.ReduceMean()
        self.cls_loss = nn.BCEWithLogitsLoss()
        self.rec_loss = nn.L1Loss("mean")

    def construct(self, img_a, att_a, att_a_, att_b, att_b_):
        """Get generator loss"""
        # generate
        zs_a = self.generator(img_a, mode="enc")
        img_fake = self.generator(zs_a, att_b_, mode="dec")
        img_recon = self.generator(zs_a, att_a_, mode="dec")

        # discriminate
        d_fake, dc_fake = self.discriminator(img_fake)

        # generator loss
        gf_loss = - self.cyc_loss(d_fake)
        gc_loss = self.cls_loss(dc_fake, att_b)
        gr_loss = self.rec_loss(img_a, img_recon)

        g_loss = gf_loss + self.lambda_2 * gc_loss + self.lambda_1 * gr_loss

        return (img_fake, g_loss, gf_loss, gc_loss, gr_loss)


class DisLoss(nn.Cell):
    """Define total discriminator loss"""

    def __init__(self, args, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

        self.cyc_loss = P.ReduceMean()
        self.cls_loss = nn.BCEWithLogitsLoss()
        self.WGANLoss = WGANGPGradientPenalty(discriminator)

        self.lambda_1 = Tensor(args.lambda_1, mstype.float32)
        self.lambda_2 = Tensor(args.lambda_2, mstype.float32)
        self.lambda_3 = Tensor(args.lambda_3, mstype.float32)
        self.lambda_gp = Tensor(args.lambda_gp, mstype.float32)

    def construct(self, img_a, att_a, att_a_, att_b, att_b_):
        """Get discriminator loss"""
        # generate
        img_fake = self.generator(img_a, att_b_, mode="enc-dec")

        # discriminate
        d_real, dc_real = self.discriminator(img_a)
        d_fake, _ = self.discriminator(img_fake)

        # discriminator losses
        d_real_loss = - self.cyc_loss(d_real)
        d_fake_loss = self.cyc_loss(d_fake)
        df_loss = d_real_loss + d_fake_loss

        df_gp = self.WGANLoss(img_a, img_fake)

        dc_loss = self.cls_loss(dc_real, att_a)

        d_loss = df_loss + self.lambda_gp * df_gp + self.lambda_3 * dc_loss

        return (d_loss, d_real_loss, d_fake_loss, dc_loss, df_gp)
