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
"""dcgan cell"""
import numpy as np
from mindspore import nn, ops
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype
from mindspore.common.initializer import Initializer
from mindspore.train.callback import ModelCheckpoint


class Reshape(nn.Cell):
    def __init__(self, shape, auto_prefix=True):
        super().__init__(auto_prefix=auto_prefix)
        self.shape = shape

    def construct(self, x):
        return ops.operations.Reshape()(x, self.shape)


class Normal(Initializer):
    """normal initializer"""
    def __init__(self, mean=0.0, sigma=0.01):
        super(Normal, self).__init__()
        self.sigma = sigma
        self.mean = mean

    def _initialize(self, arr):
        """inhert method"""
        np.random.seed(999)
        num = np.random.normal(self.mean, self.sigma, arr.shape)
        if arr.shape == ():
            arr = arr.reshape((1))
            arr[:] = num
            arr = arr.reshape(())
        else:
            if isinstance(num, np.ndarray):
                arr[:] = num[:]
            else:
                arr[:] = num


class DcganModelCheckpoint(ModelCheckpoint):
    """inherit official ModelCheckpoint"""
    def __init__(self, config, directory, prefix='dcgan'):
        super().__init__(prefix, directory, config)

    def save_ckpt(self, cb_params, force_to_save=False):
        """save ckpt"""
        super()._save_ckpt(cb_params, force_to_save)


class WithLossCellD(nn.Cell):
    """class WithLossCellD"""
    def __init__(self, netD, netG, loss_fn):
        super(WithLossCellD, self).__init__(auto_prefix=True)
        self.netD = netD
        self.netG = netG
        self.loss_fn = loss_fn

    def construct(self, real_data, latent_code):
        """class WithLossCellD construct"""
        ones = ops.Ones()
        zeros = ops.Zeros()

        out1 = self.netD(real_data)
        label1 = ones(out1.shape, mstype.float32)
        loss1 = self.loss_fn(out1, label1)

        fake_data = self.netG(latent_code)
        fake_data = F.stop_gradient(fake_data)
        out2 = self.netD(fake_data)
        label2 = zeros(out2.shape, mstype.float32)
        loss2 = self.loss_fn(out2, label2)
        return loss1 + loss2

    @property
    def backbone_network(self):
        """class WithLossCellD backbone_network"""
        return self.netD


class WithLossCellG(nn.Cell):
    """class WithLossCellG"""
    def __init__(self, netD, netG, loss_fn):
        super(WithLossCellG, self).__init__(auto_prefix=True)
        self.netD = netD
        self.netG = netG
        self.loss_fn = loss_fn

    def construct(self, latent_code):
        ones = ops.Ones()
        fake_data = self.netG(latent_code)
        out = self.netD(fake_data)
        label = ones(out.shape, mstype.float32)
        loss = self.loss_fn(out, label)
        return loss

    @property
    def backbone_network(self):
        return self.netG
