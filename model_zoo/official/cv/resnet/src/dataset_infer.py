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
create train or eval dataset.
"""
import os
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
from mindspore.communication.management import init, get_rank, get_group_size
from src.model_utils.config import config


class ImgDataset:
    """
    create img dataset.

    Args:
    Returns:
        de_dataset.
    """

    def __init__(self, dataset_path):
        super(ImgDataset, self).__init__()
        self.data = []
        self.dir_label_dict = {}
        self.img_format = (".bmp", ".png", ".jpg", ".jpeg")
        self.dir_label = config.infer_label
        dataset_list = sorted(os.listdir(dataset_path))
        file_exist = dir_exist = False
        for index, data_name in enumerate(dataset_list):
            data_path = os.path.join(dataset_path, data_name)
            if os.path.isdir(data_path):
                dir_exist = True
                self.dir_label_dict = self.get_file_label(data_name, data_path, index)
            if os.path.isfile(data_path):
                file_exist = True
                self.dir_label_dict = self.get_file_label(data_name, data_path, index=-1)
            if dir_exist and file_exist:
                raise ValueError(f"{dataset_path} can not concurrently have image file and directory")

        for data_name, img_label in self.dir_label_dict.items():
            if os.path.isfile(data_name):
                if not data_name.lower().endswith(self.img_format):
                    continue
                img_data, file_name = self.read_image_data(data_name)
                self.data.append((img_label, img_data, file_name))
            else:
                for file in os.listdir(data_name):
                    if not file.lower().endswith(self.img_format):
                        continue
                    file_path = os.path.join(data_name, file)
                    img_data, file_name = self.read_image_data(file_path)
                    self.data.append((img_label, img_data, file_name))

    def get_file_label(self, data_name, data_path, index):
        if self.dir_label and data_name not in self.dir_label:
            return self.dir_label_dict
        if self.dir_label and os.path.isdir(data_name):
            data_path_name = os.path.split(data_path)[-1]
            self.dir_label_dict[data_path] = self.dir_label[data_path_name]
        else:
            self.dir_label_dict[data_path] = index
        return self.dir_label_dict

    def read_image_data(self, file_path):
        file_name = os.path.split(file_path)[-1]
        img_data = np.fromfile(file_path, np.uint8)
        file_name = np.fromstring(file_name, np.uint8)
        file_name = np.pad(file_name, (0, 300 - file_name.shape[0]))
        return img_data, file_name

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def create_dataset(dataset_path, do_train, repeat_num=1, batch_size=32, target="Ascend", distribute=False):
    """
    create a train or eval imagenet2012 dataset for resnet50

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend
        distribute(bool): data for distribute or not. Default: False

    Returns:
        dataset
    """
    if target == "Ascend":
        device_num, rank_id = _get_rank_info()
    else:
        if distribute:
            init()
            rank_id = get_rank()
            device_num = get_group_size()
        else:
            device_num = 1

    dataset_generator = ImgDataset(dataset_path)
    if device_num == 1:
        data_set = ds.GeneratorDataset(source=dataset_generator, column_names=["label", "image", "filename"],
                                       num_parallel_workers=8, shuffle=True)
    else:
        data_set = ds.GeneratorDataset(source=dataset_generator, column_names=["label", "image", "filename"],
                                       num_parallel_workers=8, shuffle=True,
                                       num_shards=device_num, shard_id=rank_id)

    image_size = 224
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # define map operations
    if do_train:
        trans = [
            C.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            C.RandomHorizontalFlip(prob=0.5),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]
    else:
        trans = [
            C.Decode(),
            C.Resize(256),
            C.CenterCrop(image_size),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]

    type_cast_op = C2.TypeCast(mstype.int32)

    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)

    if do_train:
        data_set = data_set.project(["image", "label"])
    else:
        data_set = data_set.project(["image", "label", "filename"])

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set


def create_dataset2(dataset_path, do_train, repeat_num=1, batch_size=32, target="Ascend", distribute=False):
    """
    create a train or eval imagenet2012 dataset for resnet101
    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend
        distribute(bool): data for distribute or not. Default: False

    Returns:
        dataset
    """
    if target == "Ascend":
        device_num, rank_id = _get_rank_info()
    else:
        if distribute:
            init()
            rank_id = get_rank()
            device_num = get_group_size()
        else:
            device_num = 1
            rank_id = 1
    dataset_generator = ImgDataset(dataset_path)
    if device_num == 1:
        data_set = ds.GeneratorDataset(source=dataset_generator, column_names=["label", "image", "filename"],
                                       num_parallel_workers=8, shuffle=True)
    else:
        data_set = ds.GeneratorDataset(source=dataset_generator, column_names=["label", "image", "filename"],
                                       num_parallel_workers=8, shuffle=True,
                                       num_shards=device_num, shard_id=rank_id)
    image_size = 224
    mean = [0.475 * 255, 0.451 * 255, 0.392 * 255]
    std = [0.275 * 255, 0.267 * 255, 0.278 * 255]

    # define map operations
    if do_train:
        trans = [
            C.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            C.RandomHorizontalFlip(rank_id / (rank_id + 1)),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]
    else:
        trans = [
            C.Decode(),
            C.Resize(256),
            C.CenterCrop(image_size),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]

    type_cast_op = C2.TypeCast(mstype.int32)

    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)
    if do_train:
        data_set = data_set.project(["image", "label"])
    else:
        data_set = data_set.project(["image", "label", "filename"])
    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)
    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set


def create_dataset3(dataset_path, do_train, repeat_num=1, batch_size=32, target="Ascend", distribute=False):
    """
    create a train or eval imagenet2012 dataset for se-resnet50

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend
        distribute(bool): data for distribute or not. Default: False

    Returns:
        dataset
    """
    if target == "Ascend":
        device_num, rank_id = _get_rank_info()
    else:
        if distribute:
            init()
            rank_id = get_rank()
            device_num = get_group_size()
        else:
            device_num = 1
    dataset_generator = ImgDataset(dataset_path)
    if device_num == 1:
        data_set = ds.GeneratorDataset(source=dataset_generator, column_names=["label", "image", "filename"],
                                       num_parallel_workers=8, shuffle=True)
    else:
        data_set = ds.GeneratorDataset(source=dataset_generator, column_names=["label", "image", "filename"],
                                       num_parallel_workers=8, shuffle=True,
                                       num_shards=device_num, shard_id=rank_id)
    image_size = 224
    mean = [123.68, 116.78, 103.94]
    std = [1.0, 1.0, 1.0]

    # define map operations
    if do_train:
        trans = [
            C.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            C.RandomHorizontalFlip(prob=0.5),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]
    else:
        trans = [
            C.Decode(),
            C.Resize(292),
            C.CenterCrop(256),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]

    type_cast_op = C2.TypeCast(mstype.int32)
    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=12)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=12)
    if do_train:
        data_set = data_set.project(["image", "label"])
    else:
        data_set = data_set.project(["image", "label", "filename"])
    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = 1
        rank_id = 0

    return rank_size, rank_id
