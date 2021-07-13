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

""""GAN Loss"""

import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops
from src.vgg19.define import vgg19
from src.loss.Meanshift import MeanShift

class DiscriminatorLoss(nn.Cell):
    """Loss for discriminator"""
    def __init__(self, discriminator, generator):
        super(DiscriminatorLoss, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.adversarial_criterion = nn.BCELoss()
        ones = ops.Ones()
        zeros = ops.Zeros()
        self.real_lable = ones((16, 1), mstype.float32)
        self.fake_lable = zeros((16, 1), mstype.float32)

    def construct(self, HR_img, LR_img):
        """dloss"""
        hr = HR_img
        lr = LR_img
        # Generating fake high resolution images from real low resolution images.
        sr = self.generator(lr)
        # Let the discriminator realize that the sample is real.
        real_output = self.discriminator(hr)
        d_loss_real = self.adversarial_criterion(real_output, self.real_lable)
        # Let the discriminator realize that the sample is false.
        fake_output = self.discriminator(sr)
        d_loss_fake = self.adversarial_criterion(fake_output, self.fake_lable)
        d_loss = d_loss_fake+d_loss_real
        return  d_loss

class GeneratorLoss(nn.Cell):
    """Loss for generator"""
    def __init__(self, discriminator, generator, vgg_ckpt):
        super(GeneratorLoss, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.mse_loss = nn.MSELoss()
        self.adversarial_criterion = nn.BCELoss()
        ones = ops.Ones()
        self.real_lable = ones((16, 1), mstype.float32)
        self.meanshif = MeanShift()
        self.vgg = vgg19(vgg_ckpt)
        for p in self.meanshif.get_parameters():
            p.requires_grad = False
    def construct(self, HR_img, LR_img):
        """gloss"""
        # L2loss
        hr = HR_img
        lr = LR_img
        sr = self.generator(lr)
        L2_loss = self.mse_loss(sr, hr)

        # adversarialloss
        fake_output = self.discriminator(sr)
        adversarial_loss = self.adversarial_criterion(fake_output, self.real_lable)

        # vggloss
        hr = (hr+1.0)/2.0
        sr = (sr+1.0)/2.0
        hr = self.meanshif(hr)
        sr = self.meanshif(sr)
        hr_feat = self.vgg(hr)
        sr_feat = self.vgg(sr)
        percep_loss = self.mse_loss(hr_feat, sr_feat)

        g_loss = 0.006*percep_loss+1e-3*adversarial_loss+L2_loss
        return  g_loss
