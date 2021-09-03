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
"""dcgan dataset"""
import os
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as vision
from src.config import dcgan_imagenet_cfg


def create_dataset_imagenet(dataset_path, num_parallel_workers=None):
    """
    create a train or eval imagenet2012 dataset for dcgan

    Args:
        dataset_path(string): the path of dataset.

    Returns:
        dataset
    """

    device_num, rank_id = _get_rank_info()

    if device_num == 1:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=num_parallel_workers)
    else:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=num_parallel_workers,
                                         num_shards=device_num, shard_id=rank_id)

    assert dcgan_imagenet_cfg.image_height == dcgan_imagenet_cfg.image_width, "image_height not equal image_width"
    image_size = dcgan_imagenet_cfg.image_height

    # define map operations
    transform_img = [
        vision.Decode(),
        vision.Resize(image_size),
        vision.CenterCrop(image_size),
        vision.HWC2CHW()
    ]

    data_set = data_set.map(input_columns="image", num_parallel_workers=num_parallel_workers, operations=transform_img,
                            output_columns="image")
    data_set = data_set.map(input_columns="image", num_parallel_workers=num_parallel_workers,
                            operations=lambda x: ((x - 127.5) / 127.5).astype("float32"))
    data_set = data_set.map(
        input_columns="image",
        operations=lambda x: (
            x,
            np.random.normal(size=(dcgan_imagenet_cfg.latent_size, 1, 1)).astype("float32")
        ),
        output_columns=["image", "latent_code"],
        column_order=["image", "latent_code"],
        num_parallel_workers=num_parallel_workers
    )

    data_set = data_set.batch(dcgan_imagenet_cfg.batch_size)

    return data_set


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        from mindspore.communication.management import get_rank, get_group_size
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = rank_id = None
    return rank_size, rank_id
