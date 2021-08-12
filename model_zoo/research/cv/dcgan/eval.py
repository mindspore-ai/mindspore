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
"""dcgan eval"""
import argparse
import numpy as np
from mindspore import context, Tensor, nn, load_checkpoint

from src.config import dcgan_imagenet_cfg as cfg
from src.generator import Generator
from src.discriminator import Discriminator
from src.cell import WithLossCellD, WithLossCellG
from src.dcgan import DCGAN


def save_imgs(gen_imgs, img_url):
    """save_imgs function"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for i in range(gen_imgs.shape[0]):
        plt.subplot(4, 4, i + 1)
        gen_imgs[i] = gen_imgs[i] * 127.5 + 127.5
        perm = (1, 2, 0)
        show_imgs = np.transpose(gen_imgs[i], perm)
        sdf = show_imgs.astype(int)
        plt.imshow(sdf)
        plt.axis("off")
    plt.savefig(img_url + "/generate.png")


def load_dcgan(ckpt_url):
    """load_dcgan function"""
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
    load_checkpoint(ckpt_url, dcgan)
    netG_trained = dcgan.myTrainOneStepCellForG.network.netG
    return netG_trained


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MindSpore dcgan training')
    parser.add_argument('--device_id', type=int, default=0, help='device id of Ascend (Default: 0)')
    parser.add_argument('--img_url', type=str, default=None, help='img save path')
    parser.add_argument('--ckpt_url', type=str, default=None, help='checkpoint load path')
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(device_id=args.device_id)

    fixed_noise = Tensor(np.random.normal(size=(16, cfg.latent_size, 1, 1)).astype("float32"))

    net_G = load_dcgan(args.ckpt_url)
    fake = net_G(fixed_noise)
    print("================saving images================")
    save_imgs(fake.asnumpy(), args.img_url)
    print("================success================")
