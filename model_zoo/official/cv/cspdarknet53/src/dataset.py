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
"""
Data operations, will be used in train.py and eval.py
"""
import os

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as V_C
from PIL import Image, ImageFile
from .utils.sampler import DistributedSampler

ImageFile.LOAD_TRUNCATED_IMAGES = True

class TxtDataset():
    """
    create txt dataset.

    Args:
    Returns:
        de_dataset.
    """

    def __init__(self, root, txt_name):
        super(TxtDataset, self).__init__()
        self.imgs = []
        self.labels = []
        fin = open(txt_name, 'r')
        for line in fin:
            image_name, label = line.strip().split(' ')
            self.imgs.append(os.path.join(root, image_name))
            self.labels.append(int(label))
        fin.close()

    def __getitem__(self, item):
        img = Image.open(self.imgs[item]).convert('RGB')
        return img, self.labels[item]

    def __len__(self):
        return len(self.imgs)


def create_dataset(data_dir, image_size, per_batch_size, rank, group_size,
                   mode="train",
                   input_mode="folder",
                   root='',
                   num_parallel_workers=None,
                   shuffle=None,
                   sampler=None,
                   class_indexing=None,
                   drop_remainder=True,
                   transform=None,
                   target_transform=None):
    "create ImageNet dataset."

    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    if transform is None:
        if mode == "train":
            transform_img = [
                V_C.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
                V_C.RandomHorizontalFlip(prob=0.5),
                V_C.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4),
                V_C.Normalize(mean=mean, std=std),
                V_C.HWC2CHW()
            ]
        else:
            transform_img = [
                V_C.Decode(),
                V_C.Resize((256, 256)),
                V_C.CenterCrop(image_size),
                V_C.Normalize(mean=mean, std=std),
                V_C.HWC2CHW()
            ]
    else:
        transform_img = transform

    if target_transform is None:
        transform_label = [C.TypeCast(mstype.int32)]
    else:
        transform_label = target_transform


    if input_mode == 'folder':
        de_dataset = ds.ImageFolderDataset(data_dir, num_parallel_workers=num_parallel_workers,
                                           shuffle=shuffle, sampler=sampler, class_indexing=class_indexing,
                                           num_shards=group_size, shard_id=rank)
    else:
        dataset = TxtDataset(root, data_dir)
        sampler = DistributedSampler(dataset, rank, group_size, shuffle=shuffle)
        de_dataset = ds.GeneratorDataset(dataset, ['image', 'label'], sampler=sampler)

    de_dataset = de_dataset.map(operations=transform_img, input_columns="image",
                                num_parallel_workers=num_parallel_workers)
    de_dataset = de_dataset.map(operations=transform_label, input_columns="label",
                                num_parallel_workers=num_parallel_workers)

    columns_to_project = ['image', 'label']
    de_dataset = de_dataset.project(columns=columns_to_project)

    de_dataset = de_dataset.batch(per_batch_size, drop_remainder=drop_remainder)
    de_dataset = de_dataset.repeat(1)

    return de_dataset
