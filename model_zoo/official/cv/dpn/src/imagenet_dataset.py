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
dataset processing.
"""
from PIL import ImageFile
from mindspore.common import dtype as mstype
import mindspore.dataset as de
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as V_C


ImageFile.LOAD_TRUNCATED_IMAGES = True


def classification_dataset(data_dir, image_size, per_batch_size, max_epoch, rank, group_size,
                           mode='train',
                           num_parallel_workers=None,
                           shuffle=None,
                           sampler=None,
                           class_indexing=None,
                           transform=None,
                           target_transform=None):
    """
    A function that returns a dataset for classification. The mode of input dataset could be "folder" or "txt".
    If it is "folder", all images within one folder have the same label. If it is "txt", all paths of images
    are written into a textfile.

    Args:
        data_dir (str): Path to the root directory that contains the dataset for "input_mode="folder"".
            Or path of the textfile that contains every image's path of the dataset.
        image_size (str): Size of the input images.
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
        >>> from mindvision.common.datasets.classification import classification_dataset
        >>> # path to imagefolder directory. This directory needs to contain sub-directories which contain the images
        >>> dataset_dir = "/path/to/imagefolder_directory"
        >>> de_dataset = classification_dataset(train_data_dir, image_size=[224, 244],
        >>>                               per_batch_size=64, max_epoch=100,
        >>>                               rank=0, group_size=4)
        >>> # Path of the textfile that contains every image's path of the dataset.
        >>> dataset_dir = "/path/to/dataset/images/train.txt"
        >>> images_dir = "/path/to/dataset/images"
        >>> de_dataset = classification_dataset(train_data_dir, image_size=[224, 244],
        >>>                               per_batch_size=64, max_epoch=100,
        >>>                               rank=0, group_size=4,
        >>>                               input_mode="txt", root=images_dir)
    """
    if mode == 'eval':
        drop_remainder = False
    else:
        drop_remainder = True
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]

    std = [255 * 0.229, 255 * 0.224, 255 * 0.225]

    if transform is None:
        if mode == 'train':
            transform_img = [
                V_C.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
                V_C.RandomHorizontalFlip(prob=0.5),
                V_C.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4),
                V_C.Normalize(mean=mean, std=std),
                V_C.HWC2CHW()
            ]
        else:
            transform_img = [
                V_C.Decode(),
                V_C.Resize((256, 256)),
                V_C.CenterCrop(image_size),
                V_C.Normalize(mean=mean, std=std),
                V_C.HWC2CHW()
            ]
    else:
        transform_img = transform

    if target_transform is None:
        transform_label = [C.TypeCast(mstype.int32)]
    else:
        transform_label = target_transform

    if group_size == 1 or mode == 'eval':
        de_dataset = de.ImageFolderDataset(data_dir, num_parallel_workers=num_parallel_workers,
                                           shuffle=shuffle, sampler=sampler, class_indexing=class_indexing)
    else:
        de_dataset = de.ImageFolderDataset(data_dir, num_parallel_workers=num_parallel_workers,
                                           shuffle=shuffle, sampler=sampler, class_indexing=class_indexing,
                                           num_shards=group_size, shard_id=rank)

    de_dataset = de_dataset.map(operations=transform_img, input_columns="image", num_parallel_workers=8)
    de_dataset = de_dataset.map(operations=transform_label, input_columns="label", num_parallel_workers=8)

    columns_to_project = ["image", "label"]
    de_dataset = de_dataset.project(columns=columns_to_project)

    de_dataset = de_dataset.batch(per_batch_size, drop_remainder=drop_remainder)
    de_dataset = de_dataset.repeat(max_epoch)

    return de_dataset
