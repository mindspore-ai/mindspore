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
"""Helper functions"""
import math
import os

import numpy as np
from mindspore import load_checkpoint


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

def resume_generator(args, generator, gen_ckpt_name):
    """Restore the trained generator"""
    print("Loading the trained models from step {}...".format(args.save_interval))
    generator_path = os.path.join('output', args.experiment_name, 'checkpoint/rank0', gen_ckpt_name)
    param_generator = load_checkpoint(generator_path, generator)

    return param_generator

def resume_discriminator(args, discriminator, dis_ckpt_name):
    """Restore the trained discriminator"""
    print("Loading the trained models from step {}...".format(args.save_interval))
    discriminator_path = os.path.join('output', args.experiment_name, 'checkpoint/rank0', dis_ckpt_name)
    param_discriminator = load_checkpoint(discriminator_path, discriminator)

    return param_discriminator

def denorm(x):
    image_numpy = (np.transpose(x, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    return image_numpy
