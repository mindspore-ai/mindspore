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
""" CelebA Dataset """
import os
import multiprocessing
import numpy as np
import mindspore.dataset as de
import mindspore.dataset.vision.c_transforms as C

from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import get_rank

from PIL import Image
from .distributed_sampler import DistributedSampler
from .datasets import make_dataset


class CelebADataset:
    """ CelebA """
    def __init__(self, dir_path, mode, selected_attrs):
        self.items = make_dataset(dir_path, mode, selected_attrs)
        self.dir_path = dir_path
        self.mode = mode
        self.filename = ''

    def __getitem__(self, index):
        filename, label = self.items[index]
        image = Image.open(os.path.join(self.dir_path, 'image', filename))
        image = np.array(image.convert('RGB'))
        label = np.array(label)
        if self.mode == 'test':
            self.filename = filename
        return image, label

    def __len__(self):
        return len(self.items)

    def get_current_filename(self):
        return self.filename


class CelebADataLoader:
    """ CelebADataLoader """
    def __init__(self,
                 root,
                 mode,
                 selected_attrs,
                 crop_size=None,
                 image_size=128,
                 batch_size=64,
                 device_num=1):
        if mode not in ['train', 'test', 'val']:
            return

        mean = [0.5 * 255] * 3
        std = [0.5 * 255] * 3
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        rank = 0
        if parallel_mode in [
                ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL,
                ParallelMode.AUTO_PARALLEL
        ]:
            rank = get_rank()
        shuffle = True
        cores = multiprocessing.cpu_count()
        num_parallel_workers = min(16, int(cores / device_num))

        if mode == 'train':
            dataset = CelebADataset(root, mode, selected_attrs)
            distributed_sampler = DistributedSampler(len(dataset),
                                                     device_num,
                                                     rank,
                                                     shuffle=shuffle)
            self.dataset_size = int(len(distributed_sampler) / batch_size)

            val_set = CelebADataset(root, 'val', selected_attrs)
            self.val_dataset_size = len(val_set)

            transform = [
                C.Resize((image_size, image_size)),
                C.Normalize(mean=mean, std=std),
                C.HWC2CHW()
            ]
            if crop_size is not None:
                transform.append(C.CenterCrop(crop_size))
            val_distributed_sampler = DistributedSampler(len(val_set),
                                                         device_num,
                                                         rank,
                                                         shuffle=shuffle)
            val_dataset = de.GeneratorDataset(val_set,
                                              column_names=["image", "label"],
                                              sampler=val_distributed_sampler,
                                              num_parallel_workers=min(
                                                  32, num_parallel_workers))
            val_dataset = val_dataset.map(operations=transform,
                                          input_columns=["image"],
                                          num_parallel_workers=min(
                                              32, num_parallel_workers))
            transform.insert(0, C.RandomHorizontalFlip())
            train_dataset = de.GeneratorDataset(
                dataset,
                column_names=["image", "label"],
                sampler=distributed_sampler,
                num_parallel_workers=min(32, num_parallel_workers))
            train_dataset = train_dataset.map(operations=transform,
                                              input_columns=["image"],
                                              num_parallel_workers=min(
                                                  32, num_parallel_workers))
            train_dataset = train_dataset.batch(batch_size,
                                                drop_remainder=True)
            train_dataset = train_dataset.repeat(200)
            val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
            val_dataset = val_dataset.repeat(200)
            self.train_loader = train_dataset.create_dict_iterator()
            self.val_loader = val_dataset.create_dict_iterator()
        else:
            dataset = CelebADataset(root, mode, selected_attrs)
            self.test_set = dataset
            self.dataset_size = int(len(dataset) / batch_size)
            distributed_sampler = DistributedSampler(len(dataset),
                                                     device_num,
                                                     rank,
                                                     shuffle=shuffle)
            test_transform = [
                C.Resize((image_size, image_size)),
                C.Normalize(mean=mean, std=std),
                C.HWC2CHW()
            ]
            test_dataset = de.GeneratorDataset(dataset,
                                               column_names=["image", "label"],
                                               sampler=distributed_sampler,
                                               num_parallel_workers=min(
                                                   1, num_parallel_workers))
            test_dataset = test_dataset.map(operations=test_transform,
                                            input_columns=["image"],
                                            num_parallel_workers=min(
                                                32, num_parallel_workers))
            test_dataset = test_dataset.batch(batch_size, drop_remainder=True)
            test_dataset = test_dataset.repeat(1)

            self.test_loader = test_dataset.create_dict_iterator()

    def __len__(self):
        return self.dataset_size
