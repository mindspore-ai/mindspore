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
"""dataset"""

import random
import os
from PIL import Image
import numpy as np
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.dataset.transforms.py_transforms import Compose
from mindspore import Tensor



def omniglot_character_folders(data_path):
    '''create folders'''
    data_folder = data_path

    character_folders = [os.path.join(data_folder, family, character) \
                         for family in os.listdir(data_folder) \
                         if os.path.isdir(os.path.join(data_folder, family)) \
                         for character in os.listdir(os.path.join(data_folder, family))]
    random.seed(2)
    random.shuffle(character_folders)

    num_train = 1200
    metatrain_character_folders = character_folders[:num_train]
    metaval_character_folders = character_folders[num_train:]

    return metatrain_character_folders, metaval_character_folders


class OmniglotTask():
    """generate task"""
    # This class is for task generation for both meta training and meta testing.
    # For meta training, we use all 20 samples without valid set (empty here).
    # For meta testing, we use 1 or 5 shot samples for training, while using the same number of samples for validation.
    # If set num_samples = 20 and chracter_folders = metatrain_character_folders, we generate tasks for meta training
    # If set num_samples = 1 or 5 and chracter_folders = metatest_chracter_folders, we generate tasks for meta testing
    def __init__(self, character_folders, num_classes, train_num, test_num):
        self.character_folders = character_folders
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num

        class_folders = random.sample(self.character_folders, self.num_classes)

        labels = np.array(range(len(class_folders)))
        labels = dict(zip(class_folders, labels))
        samples = dict()

        self.train_roots = []
        self.test_roots = []
        for c in class_folders:
            temp = [os.path.join(c, x) for x in os.listdir(c)]
            samples[c] = random.sample(temp, len(temp))
            self.train_roots += samples[c][:train_num]
            self.test_roots += samples[c][train_num:train_num + test_num]
        self.train_labels = [labels[self.get_class(x)] for x in self.train_roots]
        self.test_labels = [labels[self.get_class(x)] for x in self.test_roots]

    def get_class(self, sample):
        if sample[0] == '/':
            return '/'+os.path.join(*sample.split('/')[:-1])
        return os.path.join(*sample.split('/')[:-1])


class FewShotDataset():
    '''Dataset'''
    def __init__(self, task, split='train', transform=None, target_transform=None, rotation=0, flip=False):
        self.transform = transform
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.angle = rotation
        self.flip = flip
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")


class Omniglot(FewShotDataset):
    '''Omniglot'''
    def __init__(self, *args, **kwargs):
        super(Omniglot, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('L')
        image = image.resize((28, 28), resample=Image.LANCZOS)
        image = image.rotate(self.angle)
        if self.flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        shape = image.size[0], image.size[1], 1
        image = np.array(image, np.float32, copy=False)
        image = image.reshape(shape)

        if self.transform is not None:
            image = self.transform(image)
            image = image[0]

        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label


class ClassBalancedSampler():
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, dataset, num_per_class, num_cl, num_inst, shuffle=True):
        self._index = 0
        self.dataset = dataset
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle
        self.batch_size = self.num_cl*self.num_inst

    def __iter__(self):

        return self

    def __next__(self):
        list1 = [i for i in range(self.num_inst)]
        random.seed(None)
        random.shuffle(list1)

        if self.shuffle:
            self.batch = [[i + j * self.num_inst for i in list1[:self.num_per_class]] for j in
                          range(self.num_cl)]
        else:
            self.batch = [[i + j * self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in
                          range(self.num_cl)]
        self.batch = [item for sublist in self.batch for item in sublist]

        if self.shuffle:
            random.shuffle(self.batch)

        if self._index >= len(self.dataset):
            raise StopIteration

        item_images = []
        item_labels = []
        for i in self.batch:
            item_image, item_label = self.dataset[i]
            item_images.append(item_image)
            item_labels.append(item_label)
            self._index += 1
        return (Tensor(item_images), Tensor(item_labels))

    def __len__(self):
        return 1


def get_data_loader(task, num_per_class=1, split='train', shuffle=True, rotation=0, flip=None):
    '''get dataloader'''
    mean, std = [0.92206], [0.08426]
    transform = Compose([py_vision.ToTensor(),  # numpy HWC-> Tensor CHW
                         py_vision.Normalize(mean=mean, std=std)])

    dataset = Omniglot(task, split=split, transform=transform, rotation=rotation, flip=flip)
    if split == 'train':
        loader = ClassBalancedSampler(dataset, num_per_class=num_per_class,
                                      num_cl=task.num_classes, num_inst=task.train_num, shuffle=shuffle)
    else:
        loader = ClassBalancedSampler(dataset, num_per_class=num_per_class,
                                      num_cl=task.num_classes, num_inst=task.test_num, shuffle=shuffle)
    return loader
