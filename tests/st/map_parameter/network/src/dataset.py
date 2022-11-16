# Copyright 2022 Huawei Technologies Co., Ltd
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

import os
import mindspore.dataset.vision.c_transforms as c_version
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset as ds
import mindspore.common.dtype as mstype


DATASET_PATH = "/home/workspace/mindspore_dataset/animal/mini_animal_12"
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_R_STD = 1
_G_STD = 1
_B_STD = 1


def create_dataset(epoch_size=1, batch_size=32, step_size=1, resize_height=224,
                   resize_width=224, full_batch=False, scale=1.0, rank_size=1):
    try:
        os.environ['DEVICE_ID']
    except KeyError:
        device_id = 0
        os.environ['DEVICE_ID'] = str(device_id)

    if full_batch:
        batch_size = batch_size * rank_size

    num_shards = 1
    shard_id = 0
    data_url = DATASET_PATH
    dataset = ds.ImageFolderDataset(data_url, num_parallel_workers=1, num_shards=num_shards,
                                    shard_id=shard_id, shuffle=False)

    # define map operations
    decode_op = c_version.Decode()
    c_version.Normalize(mean=[_R_MEAN, _G_MEAN, _B_MEAN], std=[_R_STD, _G_STD, _B_STD])
    random_resize_op = c_version.Resize((resize_height, resize_width))
    channelswap_op = c_version.HWC2CHW()
    rescale = scale / 255.0
    shift = 0.0
    rescale_op = c_version.Rescale(rescale, shift)
    type_cast_label = C.TypeCast(mstype.float32)
    type_cast_image = C.TypeCast(mstype.int32)

    dataset = dataset.map(input_columns="label", operations=C.OneHot(dataset.num_classes()))
    dataset = dataset.map(input_columns="label", operations=type_cast_label, num_parallel_workers=1)

    dataset = dataset.map(input_columns="image", operations=decode_op, num_parallel_workers=1)
    dataset = dataset.map(input_columns="image", operations=random_resize_op, num_parallel_workers=1)
    dataset = dataset.map(input_columns="image", operations=rescale_op, num_parallel_workers=1)
    dataset = dataset.map(input_columns="image", operations=channelswap_op, num_parallel_workers=1)
    dataset = dataset.map(input_columns="image", operations=type_cast_image, num_parallel_workers=1)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset
