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
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import mindspore.dataset as ds
from .augmentation import preproc
from .utils import bbox_encode


def TargetTransform(target):
    """Target Transform"""
    classes = {'background': 0, 'face': 1}
    keep_difficult = True
    results = []
    for obj in target.iter('object'):
        difficult = int(obj.find('difficult').text) == 1
        if not keep_difficult and difficult:
            continue
        name = obj.find('name').text.lower().strip()
        bbox = obj.find('bndbox')

        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        bndbox = []
        for _, pt in enumerate(pts):
            cur_pt = int(bbox.find(pt).text)
            bndbox.append(cur_pt)
        label_idx = classes[name]
        bndbox.append(label_idx)
        results.append(bndbox)  # [xmin, ymin, xmax, ymax, label_ind]
    return results


class WiderFaceWithVOCType():
    """WiderFaceWithVOCType"""
    def __init__(self, data_dir, target_transform=TargetTransform):
        self.data_dir = data_dir
        self.target_transform = target_transform
        self._annopath = os.path.join(self.data_dir, 'annotations', '%s')
        self._imgpath = os.path.join(self.data_dir, 'images', '%s')
        self.images_list = []
        self.labels_list = []
        with open(os.path.join(self.data_dir, 'train_img_list.txt'), 'r') as f:
            lines = f.readlines()
        for line in lines:
            img_id = line.split()[0], line.split()[1]+'.xml'
            target = ET.parse(self._annopath % img_id[1]).getroot()
            img = self._imgpath % img_id[0]

            if self.target_transform is not None:
                target = self.target_transform(target)

            self.images_list.append(img)
            self.labels_list.append(target)

    def __getitem__(self, item):
        return self.images_list[item], self.labels_list[item]

    def __len__(self):
        return len(self.images_list)


def read_dataset(img_path, annotation):
    cv2.setNumThreads(2)

    if isinstance(img_path, str):
        img = cv2.imread(img_path)
    else:
        img = cv2.imread(img_path.tostring().decode("utf-8"))

    target = np.array(annotation).astype(np.float32)
    return img, target

def create_dataset(data_dir, cfg, batch_size=32, repeat_num=1, shuffle=True, multiprocessing=True, num_worker=8):
    """create dataset"""
    dataset = WiderFaceWithVOCType(data_dir)

    if cfg['rank_size'] == 1:
        data_set = ds.GeneratorDataset(dataset, ["image", "annotation"],
                                       shuffle=shuffle,
                                       num_parallel_workers=num_worker)
    else:
        data_set = ds.GeneratorDataset(dataset, ["image", "annotation"],
                                       shuffle=shuffle,
                                       num_parallel_workers=num_worker,
                                       num_shards=cfg['rank_size'],
                                       shard_id=cfg['rank_id'])

    aug = preproc(cfg['image_size'][0])
    encode = bbox_encode(cfg)

    def union_data(image, annot):
        i, a = read_dataset(image, annot)
        i, a = aug(i, a)
        out = encode(i, a)

        return out

    data_set = data_set.map(input_columns=["image", "annotation"],
                            output_columns=["image", "truths", "conf"],
                            column_order=["image", "truths", "conf"],
                            operations=union_data,
                            python_multiprocessing=multiprocessing,
                            num_parallel_workers=num_worker)

    data_set = data_set.batch(batch_size, drop_remainder=True)
    data_set = data_set.repeat(repeat_num)


    return data_set
