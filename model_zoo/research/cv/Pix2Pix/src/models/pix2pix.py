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
    Define Pix2Pix model.
"""

import mindspore.nn as nn
from .generator_model import UnetGenerator
from .discriminator_model import Discriminator
from .init_w import init_weights
from ..utils.config import get_args

args = get_args()

class Pix2Pix(nn.Cell):
    def __init__(self, discriminator, generator):
        super(Pix2Pix, self).__init__(auto_prefix=True)
        self.netD = discriminator
        self.netG = generator

    def construct(self, realA, realB):
        fakeB = self.netG(realA)
        return fakeB

def get_generator():
    """
        Return a generator by args.
    """
    netG = UnetGenerator(in_planes=3, out_planes=3)
    init_weights(netG, init_type=args.init_type, init_gain=args.init_gain)
    return netG


def get_discriminator():
    """
        Return a discriminator by args.
    """
    netD = Discriminator(in_planes=6, ndf=64, n_layers=3)
    init_weights(netD, init_type=args.init_type, init_gain=args.init_gain)
    return netD
