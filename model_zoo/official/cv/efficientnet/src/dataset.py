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

from src.config import efficientnet_b0_config_gpu as cfg
from src.transform import RandAugment

ds.config.set_seed(cfg.random_seed)


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

img_size = (224, 224)
crop_pct = 0.875
rescale = 1.0 / 255.0
shift = 0.0
inter_method = 'bilinear'
resize_value = 224    # img_size
scale = (0.08, 1.0)
ratio = (3./4., 4./3.)
inter_str = 'bicubic'

def str2MsInter(method):
    if method == 'bicubic':
        return Inter.BICUBIC
    if method == 'nearest':
        return Inter.NEAREST
    return Inter.BILINEAR

def create_dataset(batch_size, train_data_url='', workers=8, distributed=False):
    if not os.path.exists(train_data_url):
        raise ValueError('Path not exists')
    interpolation = str2MsInter(inter_str)

    c_decode_op = C.Decode()
    type_cast_op = C2.TypeCast(mstype.int32)
    random_resize_crop_op = C.RandomResizedCrop(size=(resize_value, resize_value), scale=scale, ratio=ratio,
                                                interpolation=interpolation)
    random_horizontal_flip_op = C.RandomHorizontalFlip(0.5)

    efficient_rand_augment = RandAugment()

    image_ops = [c_decode_op, random_resize_crop_op, random_horizontal_flip_op]

    rank_id = get_rank() if distributed else 0
    rank_size = get_group_size() if distributed else 1

    dataset_train = ds.ImageFolderDataset(train_data_url,
                                          num_parallel_workers=workers,
                                          shuffle=True,
                                          num_shards=rank_size,
                                          shard_id=rank_id)
    dataset_train = dataset_train.map(input_columns=["image"],
                                      operations=image_ops,
                                      num_parallel_workers=workers)
    dataset_train = dataset_train.map(input_columns=["label"],
                                      operations=type_cast_op,
                                      num_parallel_workers=workers)
    ds_train = dataset_train.batch(batch_size,
                                   per_batch_map=efficient_rand_augment,
                                   input_columns=["image", "label"],
                                   num_parallel_workers=2,
                                   drop_remainder=True)
    return ds_train


def create_dataset_val(batch_size=128, val_data_url='', workers=8, distributed=False):
    if not os.path.exists(val_data_url):
        raise ValueError('Path not exists')
    rank_id = get_rank() if distributed else 0
    rank_size = get_group_size() if distributed else 1
    dataset = ds.ImageFolderDataset(val_data_url, num_parallel_workers=workers,
                                    num_shards=rank_size, shard_id=rank_id, shuffle=False)
    scale_size = None
    interpolation = str2MsInter(inter_method)

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
    center_crop = C.CenterCrop(size=224)
    rescale_op = C.Rescale(rescale, shift)
    normalize_op = C.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    changeswap_op = C.HWC2CHW()

    ctrans = [decode_op, resize_op, center_crop, rescale_op, normalize_op, changeswap_op]

    dataset = dataset.map(input_columns=["label"], operations=type_cast_op, num_parallel_workers=workers)
    dataset = dataset.map(input_columns=["image"], operations=ctrans, num_parallel_workers=workers)
    dataset = dataset.batch(batch_size, drop_remainder=True, num_parallel_workers=workers)
    return dataset
