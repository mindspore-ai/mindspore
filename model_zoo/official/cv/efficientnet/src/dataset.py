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
"""
Data operations, will be used in train.py and eval.py
"""
import math
import os

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.vision.c_transforms as C
from mindspore.communication.management import get_group_size, get_rank
from mindspore.dataset.vision import Inter

from src.config import basic_config, dataset_config, resize_value
from src.transform import RandAugment

ds.config.set_seed(basic_config.random_seed)


crop_pct = 0.875
rescale = 1.0 / 255.0
shift = 0.0
inter_method = 'bilinear'
scale = (0.08, 1.0)
ratio = (3./4., 4./3.)
inter_str = 'bicubic'

def str2MsInter(method):
    if method == 'bicubic':
        return Inter.BICUBIC
    if method == 'nearest':
        return Inter.NEAREST
    return Inter.BILINEAR

def create_dataset(datatype_type, train_data_url, batch_size, workers=8, distributed=False):
    if not os.path.exists(train_data_url):
        raise ValueError('Path not exists')

    interpolation = str2MsInter(inter_str)

    c_decode_op = C.Decode()
    type_cast_op = C2.TypeCast(mstype.int32)
    random_resize_crop_op = C.RandomResizedCrop(size=(resize_value, resize_value), scale=scale, ratio=ratio,
                                                interpolation=interpolation)
    random_horizontal_flip_op = C.RandomHorizontalFlip(0.5)
    efficient_rand_augment = RandAugment(dataset_config[datatype_type])

    # load dataset
    rank_id = get_rank() if distributed else 0
    rank_size = get_group_size() if distributed else 1

    if datatype_type.lower() == 'imagenet':
        dataset_train = ds.ImageFolderDataset(train_data_url,
                                              num_parallel_workers=workers,
                                              shuffle=True,
                                              num_shards=rank_size,
                                              shard_id=rank_id)
        image_ops = [c_decode_op, random_resize_crop_op, random_horizontal_flip_op]
    elif datatype_type.lower() == 'cifar10':
        dataset_train = ds.Cifar10Dataset(train_data_url,
                                          usage="train",
                                          num_parallel_workers=workers,
                                          shuffle=True,
                                          num_shards=rank_size,
                                          shard_id=rank_id)
        image_ops = [random_resize_crop_op, random_horizontal_flip_op]
    else:
        raise NotImplementedError("Only supported for ImageNet or CIFAR10 dataset")

    # build dataset
    dataset_train = dataset_train.map(input_columns=["image"],
                                      operations=image_ops,
                                      num_parallel_workers=workers)
    dataset_train = dataset_train.map(input_columns=["label"],
                                      operations=type_cast_op,
                                      num_parallel_workers=workers)
    ds_train = dataset_train.batch(batch_size,
                                   per_batch_map=efficient_rand_augment,
                                   input_columns=["image", "label"],
                                   num_parallel_workers=workers,
                                   drop_remainder=True)
    return ds_train


def create_dataset_val(datatype_type, val_data_url, batch_size=128, workers=8, distributed=False):
    if not os.path.exists(val_data_url):
        raise ValueError('Path not exists')

    interpolation = str2MsInter(inter_method)

    img_size = resize_value
    if isinstance(img_size, tuple):
        assert len(img_size) == 2
        if img_size[-1] == img_size[-2]:
            scale_size = int(math.floor(img_size[0] / crop_pct))
        else:
            scale_size = tuple([int(x / crop_pct) for x in img_size])
    else:
        scale_size = int(math.floor(img_size / crop_pct))

    type_cast_op = C2.TypeCast(mstype.int32)
    decode_op = C.Decode()
    resize_op = C.Resize(size=scale_size, interpolation=interpolation)
    center_crop = C.CenterCrop(size=resize_value)
    rescale_op = C.Rescale(rescale, shift)
    normalize_op = C.Normalize(dataset_config[datatype_type].mean, dataset_config[datatype_type].std)
    changeswap_op = C.HWC2CHW()

    # load dataset
    rank_id = get_rank() if distributed else 0
    rank_size = get_group_size() if distributed else 1

    if datatype_type.lower() == 'imagenet':
        dataset = ds.ImageFolderDataset(val_data_url, num_parallel_workers=workers,
                                        num_shards=rank_size, shard_id=rank_id, shuffle=False)
        ctrans = [decode_op, resize_op, center_crop, rescale_op, normalize_op, changeswap_op]
    elif datatype_type.lower() == 'cifar10':
        dataset = ds.Cifar10Dataset(val_data_url, usage="test", num_parallel_workers=workers,
                                    num_shards=rank_size, shard_id=rank_id, shuffle=False)
        ctrans = [resize_op, center_crop, rescale_op, normalize_op, changeswap_op]
    else:
        raise NotImplementedError("Only supported for ImageNet or CIFAR10 dataset")

    dataset = dataset.map(input_columns=["label"], operations=type_cast_op, num_parallel_workers=workers)
    dataset = dataset.map(input_columns=["image"], operations=ctrans, num_parallel_workers=workers)
    dataset = dataset.batch(batch_size, drop_remainder=True, num_parallel_workers=workers)
    return dataset
