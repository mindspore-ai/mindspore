# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Dataset utilities for animal image classification with distributed training support.

This module provides dataset creation functions for testing distributed training scenarios
with animal image data. It supports multiple dataset creation modes with varying levels
of randomness, data augmentation, and distributed sharding configurations.

Key Features:
- Image loading and preprocessing from ImageFolderDataset
- Distributed data sharding across multiple ranks
- Configurable data augmentation (resizing, flipping, normalization)
- Support for different data types (FP16, FP32, FP64, INT32, INT64)
- Dataset splitting strategies (split and unsplit modes)
- Normalization with standard ImageNet mean/std values

Functions:
- create_animal_dataset: Create augmented dataset with random operations for training
- create_animal_no_random_dataset: Create deterministic dataset with optional distributed sharding
- create_splited_data_dataset: Create dataset with spatial splitting for multi-device testing
- create_no_splited_data_dataset: Create dataset without spatial splitting

Constants:
- DATASET_PATH: Root path to the animal dataset directory
- Normalization parameters: _R_MEAN, _G_MEAN, _B_MEAN, _R_STD, _G_STD, _B_STD
"""
import os
import mindspore.dataset.vision as deMap
import mindspore.dataset.transforms as C
import mindspore.dataset as ds
import mindspore.common.dtype as mstype
from mindspore.communication.management import init
from mindspore.communication.management import get_rank
from mindspore.communication.management import get_group_size
from mindspore import log as logger


CUR_DIR = os.path.split(__file__)[0]
DATASET_PATH = os.path.join(CUR_DIR, "./animal")
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_R_STD = 1
_G_STD = 1
_B_STD = 1


def create_animal_dataset(epoch_size=2, batch_size=32, step_size=1, resize_height=224,
                          resize_width=224, label_dtype='FP32'):
    logger.info(f"MindSporeTest::epoch_size={epoch_size} and step_size={step_size} "
                f"param are unused.")
    env_dist = os.environ
    try:
        env_dist['DEVICE_ID']
    except KeyError:
        device_id = 0
        os.environ['DEVICE_ID'] = str(device_id)

    num_shards = os.getenv('RANK_SIZE', None)
    shard_id = os.getenv('RANK_ID', None)
    num_shards = int(num_shards) if num_shards is not None else None
    shard_id = int(shard_id) if shard_id is not None else None

    data_url = DATASET_PATH
    dataset = ds.ImageFolderDataset(data_url, num_shards=num_shards, shard_id=shard_id)

    # define map operations
    decode_op = deMap.Decode()
    deMap.Normalize(mean=[_R_MEAN, _G_MEAN, _B_MEAN], std=[_R_STD, _G_STD, _B_STD])
    random_resize_op = deMap.RandomResize((resize_height, resize_width))
    vertical_flip_op = deMap.RandomVerticalFlip()
    channelswap_op = deMap.HWC2CHW()
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_op = deMap.Rescale(rescale, shift)
    type_cast_op = None
    if label_dtype == 'FP32':
        type_cast_op = C.TypeCast(mstype.float32)
        dataset = dataset.map(input_columns="label", operations=C.OneHot(dataset.num_classes()))
    elif label_dtype == 'FP16':
        type_cast_op = C.TypeCast(mstype.float16)
        dataset = dataset.map(input_columns="label", operations=C.OneHot(dataset.num_classes()))
    elif label_dtype == 'FP64':
        type_cast_op = C.TypeCast(mstype.float64)
        dataset = dataset.map(input_columns="label", operations=C.OneHot(dataset.num_classes()))
    elif label_dtype == 'INT32':
        type_cast_op = C.TypeCast(mstype.int32)
    elif label_dtype == 'INT64':
        type_cast_op = C.TypeCast(mstype.int64)

    dataset = dataset.map(input_columns="label", operations=type_cast_op)
    dataset = dataset.map(input_columns="image", operations=decode_op)
    dataset = dataset.map(input_columns="image", operations=random_resize_op)
    dataset = dataset.map(input_columns="image", operations=vertical_flip_op)
    dataset = dataset.map(input_columns="image", operations=rescale_op)
    dataset = dataset.map(input_columns="image", operations=channelswap_op)

    dataset = dataset.shuffle(buffer_size=10000)  # 10000 as in imageNet train script
    dataset = dataset.batch(batch_size, drop_remainder=True)
    # tdt :4, GPU : 1
    return dataset


def create_animal_no_random_dataset(epoch_size=1, batch_size=32, step_size=1, resize_height=224,
                                    resize_width=224, full_batch=False, input_dtype='FP32',
                                    label_dtype='FP32', standalone=True, rtol=1.0):
    logger.info(f"MindSporeTest::epoch_size={epoch_size} and step_size={step_size} "
                f"param are unused.")
    env_dist = os.environ
    try:
        env_dist['DEVICE_ID']
    except KeyError:
        device_id = 0
        os.environ['DEVICE_ID'] = str(device_id)

    if standalone:
        num_shards = 1
        shard_id = 0
    else:
        init()
        if full_batch:
            num_shards = 1
            shard_id = 0
        else:
            num_shards = get_group_size()
            shard_id = get_rank()
    num_shards = int(num_shards) if num_shards is not None else None
    shard_id = int(shard_id) if shard_id is not None else None

    data_url = DATASET_PATH
    dataset = ds.ImageFolderDataset(data_url, num_parallel_workers=1, num_shards=num_shards,
                                    shard_id=shard_id, shuffle=False)

    # define map operations
    decode_op = deMap.Decode()
    deMap.Normalize(mean=[_R_MEAN, _G_MEAN, _B_MEAN], std=[_R_STD, _G_STD, _B_STD])
    random_resize_op = deMap.Resize((resize_height, resize_width))
    channelswap_op = deMap.HWC2CHW()
    rescale = rtol / 255.0
    shift = 0.0
    rescale_op = deMap.Rescale(rescale, shift)
    type_cast_op = None
    type_out = None

    if label_dtype == 'FP32':
        type_cast_op = C.TypeCast(mstype.float32)
        dataset = dataset.map(input_columns="label", operations=C.OneHot(dataset.num_classes()))
    elif label_dtype == 'FP16':
        type_cast_op = C.TypeCast(mstype.float16)
        dataset = dataset.map(input_columns="label", operations=C.OneHot(dataset.num_classes()))
    elif label_dtype == 'INT32':
        type_cast_op = C.TypeCast(mstype.int32)
    elif label_dtype == 'INT64':
        type_cast_op = C.TypeCast(mstype.int64)

    if input_dtype == 'FP32':
        type_out = C.TypeCast(mstype.float32)
    elif input_dtype == 'FP16':
        type_out = C.TypeCast(mstype.float16)
    elif input_dtype == 'INT32':
        type_out = C.TypeCast(mstype.int32)
    elif input_dtype == 'INT64':
        type_out = C.TypeCast(mstype.int64)

    dataset = dataset.map(input_columns="label", operations=type_cast_op, num_parallel_workers=1)
    dataset = dataset.map(input_columns="image", operations=decode_op, num_parallel_workers=1)
    dataset = dataset.map(input_columns="image", operations=random_resize_op,
                          num_parallel_workers=1)
    dataset = dataset.map(input_columns="image", operations=rescale_op, num_parallel_workers=1)
    dataset = dataset.map(input_columns="image", operations=channelswap_op, num_parallel_workers=1)
    dataset = dataset.map(input_columns="image", operations=type_out, num_parallel_workers=1)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    # tdt :4, GPU : 1
    return dataset


def create_splited_data_dataset(img_h=1, img_w=1, mask_h=1, mask_w=1, batch_size=64,
                                resize_height=224, resize_width=224):
    """
    created a splited dataset;num_h and num_w are used in the SlicePatches interface to
    control the splitting of H and W dimensions.

    only support fo dataset_strategy

    num_h: H dimension of dataset
    num_w: W dimension of dataset
    deMap.SlicePatches:Slice Tensor to multiple patches in horizontal and vertical directions.
    """
    data_url = DATASET_PATH

    dataset = ds.ImageFolderDataset(data_url)

    # define map operations
    decode_op = deMap.Decode()
    deMap.Normalize(mean=[_R_MEAN, _G_MEAN, _B_MEAN], std=[_R_STD, _G_STD, _B_STD])
    random_resize_op = deMap.RandomResize((resize_height, resize_width))
    vertical_flip_op = deMap.RandomVerticalFlip()
    channelswap_op = deMap.HWC2CHW()
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_op = deMap.Rescale(rescale, shift)
    slice_patches_img_op = deMap.SlicePatches(img_h, img_w)
    img_cols = ['img' + str(x) for x in range(img_h * img_w)]
    slice_patches_mask_op = deMap.SlicePatches(mask_h, mask_w)
    mask_cols = ['mask' + str(x) for x in range(mask_h * mask_w)]

    dataset = dataset.map(input_columns="image", operations=decode_op)
    dataset = dataset.map(input_columns="image", operations=random_resize_op)
    dataset = dataset.map(input_columns="image", operations=vertical_flip_op)
    dataset = dataset.map(input_columns="image", operations=rescale_op)

    dataset = dataset.map(operations=C.Duplicate(),
                          input_columns=["image"],
                          output_columns=["image", "mask"])
    dataset = dataset.project(["image", "mask"])
    dataset = dataset.map(operations=slice_patches_img_op, input_columns="image",
                          output_columns=img_cols)
    dataset = dataset.project(img_cols + ["mask"])
    dataset = dataset.map(operations=slice_patches_mask_op, input_columns="mask",
                          output_columns=mask_cols)
    dataset = dataset.project(img_cols + mask_cols)
    for col in img_cols:
        dataset = dataset.map(input_columns=col, operations=channelswap_op)

    for col in mask_cols:
        dataset = dataset.map(input_columns=col, operations=channelswap_op)

    dataset = dataset.batch(batch_size, drop_remainder=True)

    class BaseDataset:
        def __init__(self, samples):
            self.samples = samples

        def __getitem__(self, index):
            sample = self.samples[index]
            return sample[0], sample[1]

        def __len__(self):
            return len(self.samples)

    split_img_num = img_h * img_w
    split_mask_num = mask_h * mask_w
    split_num_list = [1, 2, 4, 8]
    if (split_img_num not in split_num_list) and (split_mask_num not in split_num_list):
        raise ValueError("The data set is not split. Replace it with another dataset.")
    rank_id = get_rank()
    device_id = rank_id
    img_list = []
    if split_img_num == 8:
        for x in dataset.create_tuple_iterator(output_numpy=True):
            if split_mask_num == 8:
                img_list.append([x[device_id], x[device_id - split_img_num]])
            elif split_mask_num == 4:
                if device_id >= 4:
                    img_list.append([x[device_id], x[device_id - split_img_num]])
                else:
                    img_list.append([x[device_id], x[device_id - split_mask_num]])
            elif split_mask_num == 2:
                if device_id % 2 == 0:
                    img_list.append([x[device_id], x[-2]])
                else:
                    img_list.append([x[device_id], x[-1]])
            else:
                img_list.append([x[device_id], x[-1]])
    if split_img_num == 4:
        for x in dataset.create_tuple_iterator(output_numpy=True):
            if split_mask_num == 8:
                if device_id > 4:
                    img_list.append([x[device_id - split_img_num], x[device_id - split_mask_num]])
                else:
                    img_list.append([x[device_id], x[device_id - split_mask_num]])
            elif split_mask_num == 4:
                if device_id > 4:
                    img_list.append([x[device_id - split_img_num], x[device_id - 8]])
                else:
                    img_list.append([x[device_id], x[device_id - split_mask_num]])
            elif split_mask_num == 2:
                if device_id > 4:
                    if device_id % 2 == 0:
                        img_list.append([x[device_id - split_img_num], x[-2]])
                    else:
                        img_list.append([x[device_id - split_img_num], x[-1]])
                else:
                    if device_id % 2 == 0:
                        img_list.append([x[device_id], x[-2]])
                    else:
                        img_list.append([x[device_id], x[-1]])
            else:
                if device_id > 4:
                    img_list.append([x[device_id - split_img_num], x[-1]])
                else:
                    img_list.append([x[device_id], x[-1]])
    if split_img_num == 2:
        for x in dataset.create_tuple_iterator(output_numpy=True):
            if device_id % 2 == 0:
                if split_mask_num == 8:
                    img_list.append([x[0], x[device_id]])
                elif split_mask_num == 4:
                    if device_id > 4:
                        img_list.append([x[0], x[device_id - split_mask_num]])
                    else:
                        img_list.append([x[0], x[device_id]])
                elif split_mask_num == 2:
                    if device_id % 2 == 0:
                        img_list.append([x[0], x[-2]])
                    else:
                        img_list.append([x[0], x[-1]])
                else:
                    img_list.append([x[0], x[-1]])
            else:
                if split_mask_num == 8:
                    img_list.append([x[1], x[device_id]])
                elif split_mask_num == 4:
                    if device_id > 4:
                        img_list.append([x[1], x[device_id - split_mask_num]])
                    else:
                        img_list.append([x[1], x[device_id]])
                elif split_mask_num == 2:
                    if device_id % 2 == 0:
                        img_list.append([x[1], x[-2]])
                    else:
                        img_list.append([x[1], x[-1]])
                else:
                    img_list.append([x[1], x[-1]])
    if split_img_num == 1:
        for x in dataset.create_tuple_iterator(output_numpy=True):
            if split_mask_num == 8:
                img_list.append([x[0], x[device_id + split_img_num]])
            elif split_mask_num == 4:
                if device_id >= 4:
                    img_list.append([x[0], x[device_id - split_mask_num + split_img_num]])
                else:
                    img_list.append([x[0], x[device_id + split_img_num]])
            elif split_mask_num == 2:
                if device_id % 2 == 0:
                    img_list.append([x[0], x[-2]])
                else:
                    img_list.append([x[0], x[-1]])
            else:
                img_list.append([x[0], x[1]])

    dataset = ds.GeneratorDataset(
        source=BaseDataset(img_list), column_names=["image", "mask"])

    return dataset


def create_no_splited_data_dataset(batch_size=64, resize_height=224, resize_width=224,
                                   input_dtype='FP32'):
    """
    created a no splited dataset.only support for dataset_strategy

    num_h: H dimension of dataset
    num_w: W dimension of dataset
    deMap.SlicePatches:Slice Tensor to multiple patches in horizontal and vertical directions.
    """
    data_url = DATASET_PATH

    dataset = ds.ImageFolderDataset(data_url)

    # define map operations
    decode_op = deMap.Decode()
    deMap.Normalize(mean=[_R_MEAN, _G_MEAN, _B_MEAN], std=[_R_STD, _G_STD, _B_STD])
    random_resize_op = deMap.RandomResize((resize_height, resize_width))
    vertical_flip_op = deMap.RandomVerticalFlip()
    channelswap_op = deMap.HWC2CHW()
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_op = deMap.Rescale(rescale, shift)
    type_out = None

    if input_dtype == 'FP32':
        type_out = C.TypeCast(mstype.float32)
    elif input_dtype == 'FP16':
        type_out = C.TypeCast(mstype.float16)
    elif input_dtype == 'INT32':
        type_out = C.TypeCast(mstype.int32)
    elif input_dtype == 'INT64':
        type_out = C.TypeCast(mstype.int64)

    dataset = dataset.map(input_columns="image", operations=decode_op)
    dataset = dataset.map(input_columns="image", operations=random_resize_op)
    dataset = dataset.map(input_columns="image", operations=vertical_flip_op)
    dataset = dataset.map(input_columns="image", operations=rescale_op)

    dataset = dataset.map(operations=C.Duplicate(),
                          input_columns=["image"],
                          output_columns=["image", "mask"])
    dataset = dataset.project(["image", "mask"])
    dataset = dataset.map(input_columns="image", operations=type_out, num_parallel_workers=1)
    dataset = dataset.map(input_columns="mask", operations=type_out, num_parallel_workers=1)
    dataset = dataset.map(input_columns="image", operations=channelswap_op)
    dataset = dataset.map(input_columns="mask", operations=channelswap_op)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset
