# Copyright 2020 Huawei Technologies Co., Ltd
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
"""Utils for cyclegan."""
import random
import numpy as np
from PIL import Image
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net


class ImagePool():
    """
    This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """
        Initialize the ImagePool class

        Args:
            pool_size (int): the size of image buffer, if pool_size=0, no buffer will be created.
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """
        Return an image from the pool.

        Args:
            images: the latest generated images from the generator

        Returns images Tensor from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if isinstance(images, Tensor):
            images = images.asnumpy()
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return Tensor(images)
        return_images = []
        for image in images:
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].copy()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = np.array(return_images)   # collect all the images and return
        if len(return_images.shape) != 4:
            raise ValueError("img should be 4d, but get shape {}".format(return_images.shape))
        return Tensor(return_images)


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
    img_pil.save(img_path)


def decode_image(img):
    """Decode a [1, C, H, W] Tensor to image numpy array."""
    mean = 0.5 * 255
    std = 0.5 * 255
    return (img.asnumpy()[0] * std + mean).astype(np.uint8).transpose((1, 2, 0))


def get_lr(args):
    """Learning rate generator."""
    if args.lr_policy == 'linear':
        lrs = [args.lr] * args.dataset_size * args.n_epochs
        lr_epoch = 0
        for epoch in range(args.n_epochs_decay):
            lr_epoch = args.lr * (args.n_epochs_decay - epoch) / args.n_epochs_decay
            lrs += [lr_epoch] * args.dataset_size
        lrs += [lr_epoch] * args.dataset_size * (args.max_epoch - args.n_epochs_decay - args.n_epochs)
        return Tensor(np.array(lrs).astype(np.float32))
    return args.lr


def load_ckpt(args, G_A, G_B, D_A=None, D_B=None):
    """Load parameter from checkpoint."""
    if args.G_A_ckpt is not None:
        param_GA = load_checkpoint(args.G_A_ckpt)
        load_param_into_net(G_A, param_GA)
    if args.G_B_ckpt is not None:
        param_GB = load_checkpoint(args.G_B_ckpt)
        load_param_into_net(G_B, param_GB)
    if D_A is not None and args.D_A_ckpt is not None:
        param_DA = load_checkpoint(args.D_A_ckpt)
        load_param_into_net(D_A, param_DA)
    if D_B is not None and args.D_B_ckpt is not None:
        param_DB = load_checkpoint(args.D_B_ckpt)
        load_param_into_net(D_B, param_DB)


def load_teacher_ckpt(net, ckpt_path, teacher, student):
    """Replace parameter name to teacher net and load parameter from checkpoint."""
    param = load_checkpoint(ckpt_path)
    new_param = {}
    for k, v in param.items():
        new_name = k.replace(student, teacher)
        new_param_name = v.name.replace(student, teacher)
        v.name = new_param_name
        new_param[new_name] = v
    load_param_into_net(net, new_param)
