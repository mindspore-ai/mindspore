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
""" DataLoader: CelebA"""

import os
import numpy as np
from PIL import Image

import mindspore.dataset as de
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.dataset.transforms import py_transforms

from src.utils import DistributedSampler


class Custom:
    """
    Custom data loader
    """

    def __init__(self, data_path, attr_path, selected_attrs):
        self.data_path = data_path

        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]
        images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)
        labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int)

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        transform = [py_vision.ToPIL()]
        transform.append(py_vision.Resize([128, 128]))
        transform.append(py_vision.ToTensor())
        transform.append(py_vision.Normalize(mean=mean, std=std))
        transform = py_transforms.Compose(transform)
        self.transform = transform
        self.images = np.array([images]) if images.size == 1 else images[0:]
        self.labels = np.array([labels]) if images.size == 1 else labels[0:]
        self.length = len(self.images)

    def __getitem__(self, index):
        image = np.asarray(Image.open(os.path.join(self.data_path, self.images[index])))
        att = np.asarray((self.labels[index] + 1) // 2)
        image = np.squeeze(self.transform(image))
        return image, att

    def __len__(self):
        return self.length


class CelebA:
    """
    CelebA dataset
    Input:
    data_path: Image Path
    attr_path: Attr_list Path
    image_size: Image Size
    mode: Train or Test
    selected_attrs: selected attributes
    transform: Image Processing
    """

    def __init__(self, data_path, attr_path, image_size, mode, selected_attrs, transform, split_point=182000):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.img_size = image_size

        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]

        images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)
        labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int)

        if mode == "train":
            self.images = images[:split_point]
            self.labels = labels[:split_point]
        if mode == "test":
            self.images = images[split_point:]
            self.labels = labels[split_point:]

        self.length = len(self.images)

    def __getitem__(self, index):
        image = np.asarray(Image.open(os.path.join(self.data_path, self.images[index])))
        att = np.asarray((self.labels[index] + 1) // 2)
        image = np.squeeze(self.transform(image))
        return image, att

    def __len__(self):
        return self.length


def get_loader(data_root, attr_path, selected_attrs, crop_size=170, image_size=128, mode="train", split_point=182000):
    """Build and return dataloader"""

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    transform = [py_vision.ToPIL()]
    transform.append(py_vision.CenterCrop((crop_size, crop_size)))
    transform.append(py_vision.Resize([image_size, image_size]))
    transform.append(py_vision.ToTensor())
    transform.append(py_vision.Normalize(mean=mean, std=std))
    transform = py_transforms.Compose(transform)

    dataset = CelebA(data_root, attr_path, image_size, mode, selected_attrs, transform, split_point=split_point)

    return dataset


def data_loader(img_path, attr_path, selected_attrs, mode="train", batch_size=1, device_num=1, rank=0, shuffle=True,
                split_point=182000):
    """CelebA data loader"""
    num_parallel_workers = 8

    dataset_loader = get_loader(img_path, attr_path, selected_attrs, mode=mode, split_point=split_point)
    length_dataset = len(dataset_loader)

    distributed_sampler = DistributedSampler(length_dataset, device_num, rank, shuffle=shuffle)
    dataset_column_names = ["image", "attr"]

    if device_num != 8:
        ds = de.GeneratorDataset(dataset_loader, column_names=dataset_column_names,
                                 num_parallel_workers=min(32, num_parallel_workers),
                                 sampler=distributed_sampler)
        ds = ds.batch(batch_size, num_parallel_workers=min(32, num_parallel_workers), drop_remainder=True)
    else:
        ds = de.GeneratorDataset(dataset_loader, column_names=dataset_column_names, sampler=distributed_sampler)
        ds = ds.batch(batch_size, num_parallel_workers=min(8, num_parallel_workers), drop_remainder=True)

    # ds = ds.repeat(200)

    return ds, length_dataset


def check_attribute_conflict(att_batch, att_name, att_names):
    """Check Attributes"""
    def _set(att, att_name):
        if att_name in att_names:
            att[att_names.index(att_name)] = 0.0

    att_id = att_names.index(att_name)
    for att in att_batch:
        if att_name in ['Bald', 'Receding_Hairline'] and att[att_id] != 0:
            _set(att, 'Bangs')
        elif att_name == 'Bangs' and att[att_id] != 0:
            _set(att, 'Bald')
            _set(att, 'Receding_Hairline')
        elif att_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'] and att[att_id] != 0:
            for n in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                if n != att_name:
                    _set(att, n)
        elif att_name in ['Straight_Hair', 'Wavy_Hair'] and att[att_id] != 0:
            for n in ['Straight_Hair', 'Wavy_Hair']:
                if n != att_name:
                    _set(att, n)
        elif att_name in ['Mustache', 'No_Beard'] and att[att_id] != 0:
            for n in ['Mustache', 'No_Beard']:
                if n != att_name:
                    _set(att, n)
    return att_batch


if __name__ == "__main__":

    attrs_default = [
        'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows',
        'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young'
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--attrs', dest='attrs', default=attrs_default, nargs='+', help='attributes to test')
    parser.add_argument('--data_path', dest='data_path', type=str, required=True)
    parser.add_argument('--attr_path', dest='attr_path', type=str, required=True)
    args = parser.parse_args()

    ####### Test CelebA #######
    context.set_context(device_target="Ascend")

    dataset_ce, length_ce = data_loader(args.data_path, args.attr_path, attrs_default, mode="train")
    i = 0
    for data in dataset_ce.create_dict_iterator():
        print('Number:', i, 'Value:', data["attr"], 'Type:', type(data["attr"]))
        i += 1
