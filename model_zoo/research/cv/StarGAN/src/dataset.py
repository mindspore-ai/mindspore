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
"""Data Processing for StarGAN"""
import os
import random
import multiprocessing
import numpy as np
from PIL import Image

import mindspore.dataset.vision.py_transforms as py_vision
import mindspore.dataset.transforms.py_transforms as py_transforms
import mindspore.dataset as de

from src.utils import DistributedSampler


def is_image_file(filename):
    """Judge whether it is an image"""
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tif', '.tiff']
    return any(filename.lower().endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir_path, max_dataset_size=float("inf")):
    """Return image list in dir"""
    images = []
    assert os.path.isdir(dir_path), "%s is not a valid directory" % dir_path

    for root, _, fnames in sorted(os.walk(dir_path)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


class CelebA:
    """
    This dataset class helps load celebA dataset.
    """
    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(1.0 if values[idx] == '1' else 0.0)

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, idx):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[idx]
        image = np.asarray(Image.open(os.path.join(self.image_dir, filename)))
        label = np.asarray(label)
        image = np.squeeze(self.transform(image))
        # image = Tensor(image, mstype.float32)
        # label = Tensor(label, mstype.float32)

        return image, label

    def __len__(self):
        """Return the number of images."""
        return self.num_images


class ImageFolderDataset:
    """
    This dataset class can load images from image folder.

    Args:
        data_root (str): Images root directory.
        max_dataset_size (int): Maximum number of return image paths.

    Returns:
        Image path list.
    """

    def __init__(self, data_root, transform, max_dataset_size=float("inf")):
        self.data_root = data_root
        self.transform = transform
        self.paths = sorted(make_dataset(data_root, max_dataset_size))
        self.size = len(self.paths)

    def __getitem__(self, index):
        img_path = self.paths[index % self.size]
        # image = np.array(Image.open(img_path).convert('RGB'))
        image = np.asarray(Image.open(img_path))
        return np.squeeze(self.transform(image)), os.path.split(img_path)[1]
        # return image, os.path.split(img_path)[1]

    def __len__(self):
        return self.size


def get_loader(data_root, attr_path, selected_attrs, crop_size=178, image_size=128,
               dataset='CelebA', mode='train'):
    """Build and return a data loader."""
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    transform = [py_vision.ToPIL()]
    if mode == 'train':
        transform.append(py_vision.RandomHorizontalFlip())
    transform.append(py_vision.CenterCrop(crop_size))
    transform.append(py_vision.Resize(image_size))
    transform.append(py_vision.ToTensor())
    transform.append(py_vision.Normalize(mean=mean, std=std))
    transform = py_transforms.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(data_root, attr_path, selected_attrs, transform, mode)

    elif dataset == 'RaFD':
        dataset = ImageFolderDataset(data_root, transform)
    return dataset


def dataloader(img_path, attr_path, selected_attr, dataset, mode='train',
               batch_size=1, device_num=1, rank=0, shuffle=True):
    """Get dataloader"""
    assert dataset in ['CelebA', 'RaFD']

    cores = multiprocessing.cpu_count()
    num_parallel_workers = int(cores / device_num)

    if dataset == 'CelebA':
        dataset_loader = get_loader(img_path, attr_path, selected_attr, mode=mode)
        length_dataset = len(dataset_loader)
        distributed_sampler = DistributedSampler(length_dataset, device_num, rank, shuffle=shuffle)
        dataset_column_names = ["image", "attr"]

    else:
        dataset_loader = get_loader(img_path, None, None, dataset='RaFD')
        length_dataset = len(dataset_loader)
        distributed_sampler = DistributedSampler(length_dataset, device_num, rank, shuffle=shuffle)
        dataset_column_names = ["image", "image_path"]

    if device_num != 8:
        ds = de.GeneratorDataset(dataset_loader, column_names=dataset_column_names,
                                 num_parallel_workers=min(32, num_parallel_workers),
                                 sampler=distributed_sampler)
        ds = ds.batch(batch_size, num_parallel_workers=min(32, num_parallel_workers), drop_remainder=True)

    else:
        ds = de.GeneratorDataset(dataset_loader, column_names=dataset_column_names, sampler=distributed_sampler)
        ds = ds.batch(batch_size, num_parallel_workers=min(8, num_parallel_workers), drop_remainder=True)
    if mode == 'train':
        ds = ds.repeat(200)

    return ds, length_dataset
