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
"""Dataset distributed sampler."""
from __future__ import division
import os
import math
import numpy as np

from mindspore import load_checkpoint
from mindspore import Tensor
from mindspore import dtype as mstype

from src.cell import init_weights
from src.model import Generator, Discriminator


class DistributedSampler:
    """Distributed sampler."""
    def __init__(self, dataset_size, num_replicas=None, rank=None, shuffle=False):
        if num_replicas is None:
            print("***********Setting world_size to 1 since it is not passed in ******************")
            num_replicas = 1
        if rank is None:
            print("***********Setting rank to 0 since it is not passed in ******************")
            rank = 0

        self.dataset_size = dataset_size
        self.num_replicas = num_replicas
        self.epoch = 0
        self.rank = rank
        self.num_samples = int(math.ceil(dataset_size * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # shuffle based on epoch
        if self.shuffle:
            indices = np.random.RandomState(seed=self.epoch).permutation(self.dataset_size)
            indices = indices.tolist()
            self.epoch += 1
        else:
            indices = list(range(self.dataset_size))

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank: self.total_size: self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


def resume_model(config, G, D):
    """Restore the trained generator and discriminator."""
    print('Loading the trained models from step {}...'.format(config.resume_iters))
    G_path = os.path.join(config.model_save_dir, f"Generator-0_%d.ckpt" % config.resume_iters)
    # D_path = os.path.join(config.model_save_dir, f"Net_D_%d.ckpt" % config.resume_iters)
    param_G = load_checkpoint(G_path, G)
    # param_D = load_checkpoint(D_path, D)

    return param_G, D


def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.trainable_params():
        num_params += np.prod(p.shape)
    print(model)
    print(name)
    print('The number of parameters: {}'.format(num_params))


def get_lr(init_lr, total_step, update_step, num_iters_decay):
    """Get changed learning rate."""
    lr_each_step = []
    lr = init_lr
    for i in range(total_step):
        if (i+1) % update_step == 0 and (i+1) > total_step-num_iters_decay:
            lr = lr - (init_lr / float(num_iters_decay))
        if lr < 0:
            lr = 1e-6
        lr_each_step.append(lr)

    lr_each_step = np.array(lr_each_step).astype(np.float32)

    return lr_each_step


def get_network(args):
    """Create and initial a generator and a discriminator."""

    G = Generator(args.g_conv_dim, args.c_dim, args.g_repeat_num)
    D = Discriminator(args.image_size, args.d_conv_dim, args.c_dim, args.d_repeat_num)

    init_weights(G, 'KaimingUniform', math.sqrt(5))
    init_weights(D, 'KaimingUniform', math.sqrt(5))

    print_network(G, 'Generator')
    print_network(D, 'Discriminator')

    return G, D


def create_labels(c_org, c_dim=5, selected_attrs=None):
    """Generate target domain labels for debugging and testing"""
    # Get hair color indices.
    hair_color_indices = []
    for i, attr_name in enumerate(selected_attrs):
        if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
            hair_color_indices.append(i)
    c_trg_list = []
    for i in range(c_dim):
        c_trg = c_org.copy()
        if i in hair_color_indices:
            c_trg[:, i] = 1
            for j in hair_color_indices:
                if j != i:
                    c_trg[:, j] = 0
        else:
            c_trg[:, i] = (c_trg[:, i] == 0)
        c_trg_list.append(c_trg)

    c_trg_list = Tensor(c_trg_list, mstype.float16)
    return c_trg_list

def denorm(x):
    image_numpy = (np.transpose(x, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    return image_numpy
