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
"""Face Recognition dataset."""
import os
import math
import numpy as np

import mindspore.dataset as de
import mindspore.dataset.vision.py_transforms as F
import mindspore.dataset.transforms.py_transforms as F2

from src.custom_dataset import DistributedCustomSampler, CustomDataset

__all__ = ['get_de_dataset']


def get_de_dataset(args):
    '''get_de_dataset'''
    lbl_transforms = [F.ToType(np.int32)]
    transform_label = F2.Compose(lbl_transforms)

    drop_remainder = False

    transforms = [F.ToPIL(),
                  F.RandomHorizontalFlip(),
                  F.ToTensor(),
                  F.Normalize(mean=[0.5], std=[0.5])]
    transform = F2.Compose(transforms)
    cache_path = os.path.join('cache', os.path.basename(args.data_dir), 'data_cache.pkl')
    print(cache_path)
    if not os.path.exists(os.path.dirname(cache_path)):
        os.makedirs(os.path.dirname(cache_path))
    dataset = CustomDataset(args.data_dir, cache_path, args.is_distributed)
    args.logger.info("dataset len:{}".format(dataset.__len__()))
    sampler = DistributedCustomSampler(dataset, num_replicas=args.world_size, rank=args.local_rank,
                                       is_distributed=args.is_distributed)
    de_dataset = de.GeneratorDataset(dataset, ["image", "label"], sampler=sampler)
    args.logger.info("after sampler de_dataset datasize :{}".format(de_dataset.get_dataset_size()))
    de_dataset = de_dataset.map(input_columns="image", operations=transform)
    de_dataset = de_dataset.map(input_columns="label", operations=transform_label)
    de_dataset = de_dataset.project(columns=["image", "label"])
    de_dataset = de_dataset.batch(args.per_batch_size, drop_remainder=drop_remainder)
    num_iter_per_npu = math.ceil(len(dataset) * 1.0 / args.world_size / args.per_batch_size)
    num_classes = len(dataset.classes)

    return de_dataset, num_iter_per_npu, num_classes
