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
"""get dataset loader"""
import os
import math
import cv2
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.dataset.transforms.py_transforms import Compose


def img_loader(path):
    """load image"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def list_reader(fileList):
    """read image list"""
    imgList = []
    with open(fileList, 'r') as f:
        for line in f.readlines():
            imgPath, label = line.strip().split(' ')
            imgList.append((imgPath, int(label)))
    return imgList


class ImageList:
    """
    class for load dataset
    """
    def __init__(self, root, fileList):
        self.root = root
        self.loader = img_loader
        self.imgList = list_reader(fileList)

    def __getitem__(self, index):
        imgPath, target = self.imgList[index]
        img = self.loader(os.path.join(self.root, imgPath))
        return img, target

    def __len__(self):
        return len(self.imgList)


def create_dataset(mode, data_url, data_list, batch_size, resize_size=144,
                   input_size=128, num_of_workers=8, is_distributed=False,
                   rank=0, group_size=1, seed=0):
    """
    create dataset for train or test
    """
    image_ops, shuffle, drop_last = None, None, None
    if mode == 'Train':
        shuffle = True
        drop_last = True
        image_ops = Compose([py_vision.ToPIL(),
                             py_vision.Resize(resize_size),
                             py_vision.RandomCrop(input_size),
                             py_vision.RandomHorizontalFlip(),
                             py_vision.ToTensor()])

    elif mode == 'Val':
        shuffle = False
        drop_last = False
        image_ops = Compose([py_vision.ToPIL(),
                             py_vision.Resize(resize_size),
                             py_vision.CenterCrop(input_size),
                             py_vision.ToTensor()])

    dataset_generator = ImageList(root=data_url, fileList=data_list)

    sampler = None
    if is_distributed:
        sampler = DistributedSampler(dataset=dataset_generator, rank=rank,
                                     group_size=group_size, shuffle=shuffle, seed=seed)

    dataset = ds.GeneratorDataset(dataset_generator, ["image", "label"],
                                  shuffle=shuffle, sampler=sampler,
                                  num_parallel_workers=num_of_workers)

    dataset = dataset.map(input_columns=["image"],
                          operations=image_ops,
                          num_parallel_workers=num_of_workers)

    dataset = dataset.batch(batch_size, num_parallel_workers=num_of_workers, drop_remainder=drop_last)
    dataset = dataset.repeat(1)

    return dataset


class DistributedSampler:
    """
    Distributed sampler
    """
    def __init__(self, dataset, rank, group_size, shuffle=True, seed=0):
        self.dataset = dataset
        self.rank = rank
        self.group_size = group_size
        self.dataset_length = len(self.dataset)
        self.num_samples = int(math.ceil(self.dataset_length * 1.0 / self.group_size))
        self.total_size = self.num_samples * self.group_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            self.seed = (self.seed + 1) & 0xffffffff
            np.random.seed(self.seed)
            indices = np.random.permutation(self.dataset_length).tolist()
        else:
            indices = list(range(len(self.dataset.classes)))

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank::self.group_size]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples
