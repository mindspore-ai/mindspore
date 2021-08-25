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
    Tools for Pix2Pix model.
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mindspore import Tensor
from src.utils.config import get_args

plt.switch_backend('Agg')
args = get_args()

def save_losses(G_losses, D_losses, idx):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Losses")
    plt.legend()
    plt.savefig(args.loss_show_dir+"/{}.png".format(idx))


def save_image(img, img_path):
    """Save a numpy image to the disk

    Parameters:
        img (numpy array / Tensor): image to save.
        image_path (str): the path of the image.
    """
    if isinstance(img, Tensor):
        img = decode_image(img)
    elif not isinstance(img, np.ndarray):
        raise ValueError("img should be Tensor or numpy array, but get {}".format(type(img)))

    img_pil = Image.fromarray(img)
    img_pil.save(img_path+".jpg")

def decode_image(img):
    """Decode a [1, C, H, W] Tensor to image numpy array."""
    mean = 0.5 * 255
    std = 0.5 * 255

    return (img.asnumpy()[0] * std + mean).astype(np.uint8).transpose((1, 2, 0))   # ——>（256，256，3）


def get_lr():
    """
    Linear learning-rate generator.
    Keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    """
    lrs = [args.lr] * args.dataset_size * args.n_epochs
    lr_epoch = 0
    for epoch in range(args.n_epochs_decay):
        lr_epoch = args.lr * (args.n_epochs_decay - epoch) / args.n_epochs_decay
        lrs += [lr_epoch] * args.dataset_size
    lrs += [lr_epoch] * args.dataset_size * (args.epoch_num - args.n_epochs_decay - args.n_epochs)
    return Tensor(np.array(lrs).astype(np.float32))
