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
create train or eval dataset.
"""
import os
import numpy as np

from mindspore import Tensor
from mindspore.train.model import Model
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2


def create_dataset(dataset_path, do_train, config, repeat_num=1):
    """
    create a train or eval dataset

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        config(struct): the config of train and eval in diffirent platform.
        repeat_num(int): the repeat times of dataset. Default: 1.

    Returns:
        dataset
    """
    if config.platform == "Ascend":
        rank_size = int(os.getenv("RANK_SIZE", '1'))
        rank_id = int(os.getenv("RANK_ID", '0'))
        if rank_size == 1:
            data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True)
        else:
            data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True,
                                             num_shards=rank_size, shard_id=rank_id)
    elif config.platform == "GPU":
        if do_train:
            if config.run_distribute:
                from mindspore.communication.management import get_rank, get_group_size
                data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True,
                                                 num_shards=get_group_size(), shard_id=get_rank())
            else:
                data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True)
        else:
            data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True)
    elif config.platform == "CPU":
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True)

    resize_height = config.image_height
    resize_width = config.image_width
    buffer_size = 1000

    # define map operations
    decode_op = C.Decode()
    resize_crop_op = C.RandomCropDecodeResize(resize_height, scale=(0.08, 1.0), ratio=(0.75, 1.333))
    horizontal_flip_op = C.RandomHorizontalFlip(prob=0.5)

    resize_op = C.Resize((256, 256))
    center_crop = C.CenterCrop(resize_width)
    rescale_op = C.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4)
    normalize_op = C.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                               std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    change_swap_op = C.HWC2CHW()

    if do_train:
        trans = [resize_crop_op, horizontal_flip_op, rescale_op, normalize_op, change_swap_op]
    else:
        trans = [decode_op, resize_op, center_crop, normalize_op, change_swap_op]

    type_cast_op = C2.TypeCast(mstype.int32)

    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)

    # apply shuffle operations
    data_set = data_set.shuffle(buffer_size=buffer_size)

    # apply batch operations
    data_set = data_set.batch(config.batch_size, drop_remainder=True)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set


def extract_features(net, dataset_path, config):
    features_folder = dataset_path + '_features'
    if not os.path.exists(features_folder):
        os.makedirs(features_folder)
    dataset = create_dataset(dataset_path=dataset_path,
                             do_train=False,
                             config=config)
    step_size = dataset.get_dataset_size()
    if step_size == 0:
        raise ValueError("The step_size of dataset is zero. Check if the images count of train dataset is more \
            than batch_size in config.py")

    model = Model(net)

    for i, data in enumerate(dataset.create_dict_iterator(output_numpy=True)):
        features_path = os.path.join(features_folder, f"feature_{i}.npy")
        label_path = os.path.join(features_folder, f"label_{i}.npy")
        if not os.path.exists(features_path) or not os.path.exists(label_path):
            image = data["image"]
            label = data["label"]
            features = model.predict(Tensor(image))
            np.save(features_path, features.asnumpy())
            np.save(label_path, label)
        print(f"Complete the batch {i + 1}/{step_size}")
    return step_size
