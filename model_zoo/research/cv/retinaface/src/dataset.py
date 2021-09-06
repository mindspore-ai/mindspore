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
"""Dataset for train and eval."""
import os
import copy
import cv2
import numpy as np

import mindspore.dataset as de
from mindspore.communication.management import init, get_rank, get_group_size

from .augmemtation import preproc
from .utils import bbox_encode


class WiderFace():
    """WiderFace"""
    def __init__(self, label_path):
        self.images_list = []
        self.labels_list = []
        f = open(label_path, 'r')
        lines = f.readlines()
        First = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if First is True:
                    First = False
                else:
                    c_labels = copy.deepcopy(labels)
                    self.labels_list.append(c_labels)
                    labels.clear()
                # remove '# '
                path = line[2:]
                path = label_path.replace('label.txt', 'images/') + path

                assert os.path.exists(path), 'image path is not exists.'

                self.images_list.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)
        # add the last label
        self.labels_list.append(labels)

        # del bbox which width is zero or height is zero
        for i in range(len(self.labels_list) - 1, -1, -1):
            labels = self.labels_list[i]
            for j in range(len(labels) - 1, -1, -1):
                label = labels[j]
                if label[2] <= 0 or label[3] <= 0:
                    labels.pop(j)
            if not labels:
                self.images_list.pop(i)
                self.labels_list.pop(i)
            else:
                self.labels_list[i] = labels

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, item):
        return self.images_list[item], self.labels_list[item]

def read_dataset(img_path, annotation):
    """read_dataset"""
    cv2.setNumThreads(2)

    if isinstance(img_path, str):
        img = cv2.imread(img_path)
    else:
        img = cv2.imread(img_path.tostring().decode("utf-8"))

    labels = annotation
    anns = np.zeros((0, 15))
    if labels.shape[0] <= 0:
        return anns
    for _, label in enumerate(labels):
        ann = np.zeros((1, 15))

        # get bbox
        ann[0, 0:2] = label[0:2]  # x1, y1
        ann[0, 2:4] = label[0:2] + label[2:4]  # x2, y2

        # get landmarks
        ann[0, 4:14] = label[[4, 5, 7, 8, 10, 11, 13, 14, 16, 17]]

        # set flag
        if (ann[0, 4] < 0):
            ann[0, 14] = -1
        else:
            ann[0, 14] = 1

        anns = np.append(anns, ann, axis=0)
    target = np.array(anns).astype(np.float32)

    return img, target


def create_dataset(data_dir, cfg, batch_size=32, repeat_num=1, shuffle=True, multiprocessing=True, num_worker=16):
    """create_dataset"""
    dataset = WiderFace(data_dir)

    if cfg['name'] == 'ResNet50':
        device_num, rank_id = _get_rank_info()
    elif cfg['name'] == 'MobileNet025':
        init("nccl")
        rank_id = get_rank()
        device_num = get_group_size()
    if device_num == 1:
        de_dataset = de.GeneratorDataset(dataset, ["image", "annotation"],
                                         shuffle=shuffle,
                                         num_parallel_workers=num_worker)
    else:
        de_dataset = de.GeneratorDataset(dataset, ["image", "annotation"],
                                         shuffle=shuffle,
                                         num_parallel_workers=num_worker,
                                         num_shards=device_num,
                                         shard_id=rank_id)

    aug = preproc(cfg['image_size'])
    encode = bbox_encode(cfg)

    def union_data(image, annot):
        i, a = read_dataset(image, annot)
        i, a = aug(i, a)
        out = encode(i, a)

        return out

    de_dataset = de_dataset.map(input_columns=["image", "annotation"],
                                output_columns=["image", "truths", "conf", "landm"],
                                column_order=["image", "truths", "conf", "landm"],
                                operations=union_data,
                                python_multiprocessing=multiprocessing,
                                num_parallel_workers=num_worker)

    de_dataset = de_dataset.batch(batch_size, drop_remainder=True)
    de_dataset = de_dataset.repeat(repeat_num)


    return de_dataset


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = rank_id = None

    return rank_size, rank_id
