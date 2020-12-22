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
"""Cycle GAN dataset."""
import os
import multiprocessing

import mindspore.dataset as de
import mindspore.dataset.vision.c_transforms as C
from .distributed_sampler import DistributedSampler
from .datasets import UnalignedDataset, ImageFolderDataset

def create_dataset(args):
    """Create dataset"""
    dataroot = args.dataroot
    phase = args.phase
    batch_size = args.batch_size
    device_num = args.device_num
    rank = args.rank
    shuffle = args.use_random
    max_dataset_size = args.max_dataset_size
    cores = multiprocessing.cpu_count()
    num_parallel_workers = min(8, int(cores / device_num))
    image_size = args.image_size
    mean = [0.5 * 255] * 3
    std = [0.5 * 255] * 3
    if phase == "train":
        dataset = UnalignedDataset(dataroot, phase, max_dataset_size=max_dataset_size, use_random=args.use_random)
        distributed_sampler = DistributedSampler(len(dataset), device_num, rank, shuffle=shuffle)
        ds = de.GeneratorDataset(dataset, column_names=["image_A", "image_B"],
                                 sampler=distributed_sampler, num_parallel_workers=num_parallel_workers)
        if args.use_random:
            trans = [
                C.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.75, 1.333)),
                C.RandomHorizontalFlip(prob=0.5),
                C.Normalize(mean=mean, std=std),
                C.HWC2CHW()
            ]
        else:
            trans = [
                C.Resize((image_size, image_size)),
                C.Normalize(mean=mean, std=std),
                C.HWC2CHW()
            ]
        ds = ds.map(operations=trans, input_columns=["image_A"], num_parallel_workers=num_parallel_workers)
        ds = ds.map(operations=trans, input_columns=["image_B"], num_parallel_workers=num_parallel_workers)
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.repeat(1)
    else:
        datadir = os.path.join(dataroot, args.data_dir)
        dataset = ImageFolderDataset(datadir, max_dataset_size=max_dataset_size)
        ds = de.GeneratorDataset(dataset, column_names=["image", "image_name"],
                                 num_parallel_workers=num_parallel_workers)
        trans = [
            C.Resize((image_size, image_size)),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]
        ds = ds.map(operations=trans, input_columns=["image"], num_parallel_workers=num_parallel_workers)
        ds = ds.batch(1, drop_remainder=True)
        ds = ds.repeat(1)
    args.dataset_size = len(dataset)
    return ds
