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
"""data process"""
import math
import sys
import os
from collections import defaultdict
import numpy as np
from PIL import ImageFile
import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True
__all__ = ['DistributedPKSampler', 'Dataset']
IMG_EXTENSIONS = ('.jpg', 'jpeg', '.png', '.ppm', '.bmp', 'pgm', '.tif', '.tiff', 'webp')


class DistributedPKSampler:
    '''DistributedPKSampler'''
    def __init__(self, dataset, shuffle=True, p=5, k=2):
        assert isinstance(dataset, PKDataset), 'PK Sampler Only Supports PK Dataset!'
        self.p = p
        self.k = k
        self.dataset = dataset
        self.epoch = 0
        self.step_nums = int(math.ceil(len(self.dataset.classes)*1.0/p))
        self.total_ids = self.step_nums*p
        self.batch_size = p*k
        self.num_samples = self.total_ids * self.k
        self.shuffle = shuffle
        self.epoch_gen = 1


    def _sample_pk(self, indices):
        '''sample pk'''
        sampled_pk = []
        for indice in indices:
            sampled_id = indice
            replacement = False
            if len(self.dataset.id2range[sampled_id]) < self.k:
                replacement = True
            index_list = np.random.choice(self.dataset.id2range[sampled_id][0:], self.k, replace=replacement)
            sampled_pk.extend(index_list.tolist())

        return sampled_pk



    def __iter__(self):
        if self.shuffle:
            self.epoch_gen = (self.epoch_gen + 1) & 0xffffffff
            np.random.seed(self.epoch_gen)
            indices = np.random.permutation(len(self.dataset.classes))
            indices = indices.tolist()
        else:
            indices = list(range(len(self.dataset.classes)))
        indices += indices[:(self.total_ids - len(indices))]
        assert len(indices) == self.total_ids

        sampled_idxs = self._sample_pk(indices)

        return iter(sampled_idxs)



    def __len__(self):
        return self.num_samples



    def set_epoch(self, epoch):
        self.epoch = epoch


def has_file_allowed_extension(filename, extensions):
    """ check if a file has an allowed extensio n.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions allowed (lowercase)

    Returns:
        bool: True if the file ends with one of the given extensions
    """
    return filename.lower().endswith(extensions)


def make_dataset(dir_name, class_to_idx, extensions=None, is_valid_file=None):
    '''make dataset'''
    images = []
    dir_name = os.path.expanduser(dir_name)
    if not (extensions is None) ^ (is_valid_file is None):
        raise ValueError("Extensions and is_valid_file should not be the same.")
    def is_valid(x):
        if extensions is not None:
            return has_file_allowed_extension(x, extensions)
        return is_valid_file(x)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir_name, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid(path):
                    item = (path, class_to_idx[target], 0.6)
                    images.append(item)
    return images


class ImageFolderPKDataset:
    '''ImageFolderPKDataset'''
    def __init__(self, root):
        self.classes, self.classes_to_idx = self._find_classes(root)
        self.samples = make_dataset(root, self.classes_to_idx, IMG_EXTENSIONS, None)
        self.id2range = self._build_id2range()
        self.all_image_idxs = range(len(self.samples))
        self.classes = list(self.id2range.keys())

    def _find_classes(self, dir_name):
        """
        Finds the class folders in a dataset

        Args:
            dir (string): root directory path

        Returns:
            tuple (class, class_to_idx): where classes are relative to dir, and class_to_idx is a directionaty

        Ensures:
            No class is a subdirectory of others
        """

        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir_name) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx


    def _build_id2range(self):
        '''map id to range'''
        id2range = defaultdict(list)
        ret_range = defaultdict(list)
        for idx, sample in enumerate(self.samples):
            label = sample[1]
            id2range[label].append((sample, idx))
        for key in id2range:
            id2range[key].sort(key=lambda x: int(os.path.basename(x[0][0]).split(".")[0]))
            for item in id2range[key]:
                ret_range[key].append(item[1])
        return ret_range


    def __getitem__(self, index):
        return self.samples[index]


    def __len__(self):
        return len(self.samples)


def pil_loader(path):
    '''pil loader'''
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class Dataset:
    '''Dataset'''
    def __init__(self, root, loader=pil_loader):
        self.dataset = ImageFolderPKDataset(root)
        print('Dataset len(dataset):{}'.format(len(self.dataset)))
        self.loader = loader
        self.classes = self.dataset.classes
        self.id2range = self.dataset.id2range


    def __getitem__(self, index):
        path, target1, target2 = self.dataset[index]
        sample = self.loader(path)
        return sample, target1, target2


    def __len__(self):
        return len(self.dataset)
