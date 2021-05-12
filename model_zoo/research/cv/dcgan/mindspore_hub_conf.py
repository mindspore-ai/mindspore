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
"""hub config."""
from mindspore import nn

from src.cell import WithLossCellD, WithLossCellG
from src.dcgan import DCGAN
from src.discriminator import Discriminator
from src.generator import Generator
from src.config import dcgan_imagenet_cfg as cfg


def create_network(name):
    """create_network function"""
    if name == "dcgan":
        netD = Discriminator()
        netG = Generator()

        criterion = nn.BCELoss(reduction='mean')

        netD_with_criterion = WithLossCellD(netD, netG, criterion)
        netG_with_criterion = WithLossCellG(netD, netG, criterion)

        optimizerD = nn.Adam(netD.trainable_params(), learning_rate=cfg.learning_rate, beta1=cfg.beta1)
        optimizerG = nn.Adam(netG.trainable_params(), learning_rate=cfg.learning_rate, beta1=cfg.beta1)

        myTrainOneStepCellForD = nn.TrainOneStepCell(netD_with_criterion, optimizerD)
        myTrainOneStepCellForG = nn.TrainOneStepCell(netG_with_criterion, optimizerG)

        dcgan = DCGAN(myTrainOneStepCellForD, myTrainOneStepCellForG)
        return dcgan
    raise NotImplementedError(f"{name} is not implemented in the repo")
