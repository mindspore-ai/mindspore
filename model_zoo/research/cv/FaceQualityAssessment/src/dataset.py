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
"""Face Quality Assessment dataset."""
import math
import warnings
import numpy as np
from PIL import Image, ImageFile

import mindspore.dataset as ds
import mindspore.dataset.vision.py_transforms as F

warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MdFaceDataset():
    """Face Landmarks dataset."""
    def __init__(self, imlist,
                 img_shape=(96, 96),
                 heatmap_shape=(48, 48)):

        self.imlist = imlist
        self.img_shape = img_shape
        self.heatmap_shape = heatmap_shape
        print('Reading data...')
        with open(imlist) as fr:
            self.imgs_info = fr.readlines()

    def _trans_cor(self, landmark, x_length, y_length):
        '''_trans_cor'''
        landmark = list(map(float, landmark))
        landmark = np.array(landmark).reshape((5, 2))
        landmark_class_label = []
        for _, cor in enumerate(landmark):
            x, y = cor
            if x < 0:
                heatmap_label = -1
            else:
                x = float(x) / float(x_length) * 96.
                y = float(y) / float(y_length) * 96.
                x_out = int(x * 1.0 * self.heatmap_shape[1] / self.img_shape[1])
                y_out = int(y * 1.0 * self.heatmap_shape[0] / self.img_shape[0])
                heatmap_label = y_out * self.heatmap_shape[1] + x_out
                if heatmap_label >= self.heatmap_shape[0]*self.heatmap_shape[1] or heatmap_label < 0:
                    heatmap_label = -1
            landmark_class_label.append(heatmap_label)
        return landmark_class_label

    def __len__(self):
        return len(self.imgs_info)

    def __getitem__(self, idx):
        path_label_info = self.imgs_info[idx].strip().split('\t')
        impath = path_label_info[0]
        image = Image.open(impath).convert('RGB')
        x_length = image.size[0]
        y_length = image.size[1]
        image = image.resize((96, 96))
        landmarks = self._trans_cor(path_label_info[4:14], x_length, y_length)
        eulers = np.array([e / 90. for e in list(map(float, path_label_info[1:4]))])
        labels = np.concatenate([eulers, landmarks], axis=0)
        sample = F.ToTensor()(image)

        return sample, labels


class DistributedSampler():
    '''DistributedSampler'''
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
        indices = indices[self.rank::self.group_size]
        return iter(indices)

    def __len__(self):
        return self.num_samples


def faceqa_dataset(imlist, per_batch_size, local_rank, world_size):
    '''faceqa dataset'''
    dataset = MdFaceDataset(imlist)
    sampler = DistributedSampler(dataset, local_rank, world_size)
    de_dataset = ds.GeneratorDataset(dataset, ["image", "label"], sampler=sampler, num_parallel_workers=16,
                                     python_multiprocessing=True)
    de_dataset = de_dataset.batch(per_batch_size, drop_remainder=True, num_parallel_workers=4)

    return de_dataset
