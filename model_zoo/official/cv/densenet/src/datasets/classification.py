# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
A function that returns a dataset for classification.
"""

import os
from PIL import Image, ImageFile
from mindspore import dtype as mstype
import mindspore.dataset as de
import mindspore.dataset.vision.c_transforms as vision_C
import mindspore.dataset.transforms.c_transforms as normal_C
from src.datasets.sampler import DistributedSampler

ImageFile.LOAD_TRUNCATED_IMAGES = True

class TxtDataset():
    """
    read dataset from txt
    """
    def __init__(self, root, txt_name):
        super(TxtDataset, self).__init__()
        self.imgs = []
        self.labels = []
        fin = open(txt_name, "r")
        for line in fin:
            img_name, label = line.strip().split(' ')
            self.imgs.append(os.path.join(root, img_name))
            self.labels.append(int(label))
        fin.close()

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        return img, self.labels[index]

    def __len__(self):
        return len(self.imgs)


def classification_dataset_imagenet(data_dir, image_size, per_batch_size, max_epoch, rank, group_size, mode='train',
                                    input_mode='folder', root='', num_parallel_workers=None, shuffle=None,
                                    sampler=None, class_indexing=None, drop_remainder=True, transform=None,
                                    target_transform=None):
    """
    A function that returns a dataset for classification. The mode of input dataset could be "folder" or "txt".
    If it is "folder", all images within one folder have the same label. If it is "txt", all paths of images
    are written into a textfile.

    Args:
        data_dir (str): Path to the root directory that contains the dataset for "input_mode="folder"".
            Or path of the textfile that contains every image's path of the dataset.
        image_size (Union(int, sequence)): Size of the input images.
        per_batch_size (int): the batch size of evey step during training.
        max_epoch (int): the number of epochs.
        rank (int): The shard ID within num_shards (default=None).
        group_size (int): Number of shards that the dataset should be divided
            into (default=None).
        mode (str): "train" or others. Default: " train".
        input_mode (str): The form of the input dataset. "folder" or "txt". Default: "folder".
        root (str): the images path for "input_mode="txt"". Default: " ".
        num_parallel_workers (int): Number of workers to read the data. Default: None.
        shuffle (bool): Whether or not to perform shuffle on the dataset
            (default=None, performs shuffle).
        sampler (Sampler): Object used to choose samples from the dataset. Default: None.
        class_indexing (dict): A str-to-int mapping from folder name to index
            (default=None, the folder names will be sorted
            alphabetically and each class will be given a
            unique index starting from 0).

    Examples:
        >>> from src.datasets.classification import classification_dataset_imagenet
        >>> # path to imagefolder directory. This directory needs to contain sub-directories which contain the images
        >>> data_dir = "/path/to/imagefolder_directory"
        >>> de_dataset = classification_dataset(data_dir, image_size=[224, 244],
        >>>                               per_batch_size=64, max_epoch=100,
        >>>                               rank=0, group_size=4)
        >>> # Path of the textfile that contains every image's path of the dataset.
        >>> data_dir = "/path/to/dataset/images/train.txt"
        >>> images_dir = "/path/to/dataset/images"
        >>> de_dataset = classification_dataset(data_dir, image_size=[224, 244],
        >>>                               per_batch_size=64, max_epoch=100,
        >>>                               rank=0, group_size=4,
        >>>                               input_mode="txt", root=images_dir)
    """

    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    if transform is None:
        if mode == 'train':
            transform_img = [
                vision_C.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
                vision_C.RandomHorizontalFlip(prob=0.5),
                vision_C.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4),
                vision_C.Normalize(mean=mean, std=std),
                vision_C.HWC2CHW()
            ]
        else:
            transform_img = [
                vision_C.Decode(),
                vision_C.Resize((256, 256)),
                vision_C.CenterCrop(image_size),
                vision_C.Normalize(mean=mean, std=std),
                vision_C.HWC2CHW()
            ]
    else:
        transform_img = transform

    if target_transform is None:
        transform_label = [
            normal_C.TypeCast(mstype.int32)
        ]
    else:
        transform_label = target_transform

    if input_mode == 'folder':
        de_dataset = de.ImageFolderDataset(data_dir, num_parallel_workers=num_parallel_workers,
                                           shuffle=shuffle, sampler=sampler, class_indexing=class_indexing,
                                           num_shards=group_size, shard_id=rank)
    else:
        dataset = TxtDataset(root, data_dir)
        sampler = DistributedSampler(dataset, rank, group_size, shuffle=shuffle)
        de_dataset = de.GeneratorDataset(dataset, ["image", "label"], sampler=sampler)

    de_dataset = de_dataset.map(input_columns="image", num_parallel_workers=8, operations=transform_img)
    de_dataset = de_dataset.map(input_columns="label", num_parallel_workers=8, operations=transform_label)

    columns_to_project = ["image", "label"]
    de_dataset = de_dataset.project(columns=columns_to_project)

    de_dataset = de_dataset.batch(per_batch_size, drop_remainder=drop_remainder)
    de_dataset = de_dataset.repeat(1)

    return de_dataset


def classification_dataset_cifar10(data_dir, image_size, per_batch_size, max_epoch, rank, group_size, mode='train',
                                   num_parallel_workers=None, shuffle=None, sampler=None, drop_remainder=True,
                                   transform=None, target_transform=None):
    """
    A function that returns cifar10 dataset for classification.

    Args:
        data_dir (str): Path to the root directory that contains the dataset's bin files.
        image_size (Union(int, sequence)): Size of the input images.
        per_batch_size (int): the batch size of evey step during training.
        max_epoch (int): the number of epochs.
        rank (int): The shard ID within num_shards (default=None).
        group_size (int): Number of shards that the dataset should be divided
            into (default=None).
        mode (str): "train" or others. Default: " train".
        input_mode (str): The form of the input dataset. "folder" or "txt". Default: "folder".
        root (str): the images path for "input_mode="txt"". Default: " ".
        num_parallel_workers (int): Number of workers to read the data. Default: None.
        shuffle (bool): Whether or not to perform shuffle on the dataset
            (default=None, performs shuffle).
        sampler (Sampler): Object used to choose samples from the dataset. Default: None.

    Examples:
        >>> from src.datasets.classification import classification_dataset_cifar10
        >>> # path to imagefolder directory. This directory needs to contain bin files of data.
        >>> data_dir = "/path/to/datafolder_directory"
        >>> de_dataset = classification_dataset_cifar10(data_dir, image_size=[32, 32],
        >>>                               per_batch_size=64, max_epoch=100,
        >>>                               rank=0, group_size=1)
    """

    mean = [0.5 * 255, 0.5 * 255, 0.5 * 255]
    std = [0.5 * 255, 0.5 * 255, 0.5 * 255]

    if transform is None:
        if mode == 'train':
            transform_img = [
                vision_C.RandomCrop(image_size, padding=4),
                vision_C.RandomHorizontalFlip(prob=0.5),
                vision_C.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4),
                vision_C.Normalize(mean=mean, std=std),
                vision_C.HWC2CHW()
            ]
        else:
            transform_img = [
                vision_C.Normalize(mean=mean, std=std),
                vision_C.HWC2CHW()
            ]
    else:
        transform_img = transform

    if target_transform is None:
        transform_label = [
            normal_C.TypeCast(mstype.int32)
        ]
    else:
        transform_label = target_transform

    de_dataset = de.Cifar10Dataset(data_dir, num_parallel_workers=num_parallel_workers, shuffle=shuffle,
                                                                      sampler=sampler, num_shards=group_size,
                                                                      shard_id=rank)

    de_dataset = de_dataset.map(input_columns="image", num_parallel_workers=8, operations=transform_img)
    de_dataset = de_dataset.map(input_columns="label", num_parallel_workers=8, operations=transform_label)

    columns_to_project = ["image", "label"]
    de_dataset = de_dataset.project(columns=columns_to_project)

    de_dataset = de_dataset.batch(per_batch_size, drop_remainder=drop_remainder)
    de_dataset = de_dataset.repeat(1)

    return de_dataset
