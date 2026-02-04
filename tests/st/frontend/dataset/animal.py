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
"""Dataset of animal."""
import os
import logging
import mindspore.dataset as ds
from mindspore import dtype as mstype
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as deMap

logger = logging.getLogger(__name__)
CUR_DIR = os.path.split(__file__)[0]
DATASET_PATH = os.path.join(CUR_DIR, "data/animal")
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_R_STD = 1
_G_STD = 1
_B_STD = 1


def create_animal_dataset(epoch_size=2, batch_size=32, step_size=1, resize_height=224,
                          resize_width=224, label_dtype='FP32'):
    logger.info("Animal Dataset::epoch_size=%d and step_size=%d param are unused.",
                epoch_size, step_size)

    data_url = DATASET_PATH
    dataset = ds.ImageFolderDataset(data_url, num_shards=None, shard_id=None)

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
    logger.info("Animal No Random Dataset::epoch_size=%d and step_size=%d param are unused.",
                epoch_size, step_size)

    num_shards = 1
    shard_id = 0
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

    type_out = None
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
