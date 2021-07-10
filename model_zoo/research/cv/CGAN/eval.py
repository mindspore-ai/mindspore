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
"""eval cgan"""
import os
import itertools
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mindspore import Tensor
from mindspore import context
from mindspore.common import dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.model import Generator

def preLauch():
    """parse the console argument"""
    parser = argparse.ArgumentParser(description='MindSpore cgan training')
    parser.add_argument('--device_id', type=int, default=0,
                        help='device id of Ascend (Default: 0)')
    parser.add_argument('--ckpt_dir', type=str,
                        default='ckpt', help='checkpoint dir of CGAN')
    parser.add_argument('--img_out', type=str,
                        default='img_eval', help='the dir of output img')
    args = parser.parse_args()

    context.set_context(device_id=args.device_id,
                        mode=context.GRAPH_MODE,
                        device_target="Ascend")
    # if not exists 'img_out', make it
    if not os.path.exists(args.img_out):
        os.mkdir(args.img_out)
    return args


def main():
    # before training, we should set some arguments
    args = preLauch()

    # training argument
    input_dim = 100

    # create G Cell & D Cell
    netG = Generator(input_dim)

    latent_code_eval = Tensor(np.random.randn(200, input_dim), dtype=mstype.float32)

    label_eval = np.zeros((200, 10))
    for i in range(200):
        j = i // 20
        label_eval[i][j] = 1
    label_eval = Tensor(label_eval, dtype=mstype.float32)

    fig, ax = plt.subplots(10, 20, figsize=(10, 5))
    for digit, num in itertools.product(range(10), range(20)):
        ax[digit, num].get_xaxis().set_visible(False)
        ax[digit, num].get_yaxis().set_visible(False)

    param_G = load_checkpoint(args.ckpt_dir)
    load_param_into_net(netG, param_G)
    gen_imgs_eval = netG(latent_code_eval, label_eval)
    for i in range(200):
        if (i + 1) % 20 == 0:
            print("process ========= {}/200".format(i+1))
        digit = i // 20
        num = i % 20
        img = gen_imgs_eval[i].asnumpy().reshape((28, 28))
        ax[digit, num].cla()
        ax[digit, num].imshow(img * 127.5 + 127.5, cmap="gray")

    label = 'eval result'
    fig.text(0.5, 0.01, label, ha='center')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    print("===========saving image===========")
    plt.savefig("./img_eval/result.png")
    print("===========success================")


if __name__ == '__main__':
    main()
