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

"""dataloader of cityscapes."""

import random
import numpy as np
from utils.serialization import json_load
from .base_dataset import BaseDataset

DEFAULT_INFO_PATH_19 = 'dataset/cityscapes_list/info.json'
DEFAULT_INFO_PATH_16 = 'dataset/cityscapes_list/info_16.json'


class CityscapesDataSet(BaseDataset):
    """dataloader of cityscapes"""
    def __init__(self, root, list_path, num_classes=19, set_name="val",
                 max_iters=None,
                 crop_size=(321, 321), mean=(128, 128, 128),
                 load_labels=True, semi=False, num_semi=100,
                 info_path=DEFAULT_INFO_PATH_19, labels_size=None, trans_img=False, del_xl=False):
        super().__init__(root, list_path, set_name, max_iters, crop_size, labels_size, mean, semi=semi,
                         num_semi=num_semi, trans_img=trans_img, del_xl=del_xl)
        self.semi = semi
        self.load_labels = load_labels

        if num_classes == 19:
            self.info = json_load(DEFAULT_INFO_PATH_19)
        elif num_classes == 16:
            self.info = json_load(DEFAULT_INFO_PATH_16)

        self.class_names = np.array(self.info['label'], dtype=np.str)
        self.mapping = np.array(self.info['label2train'], dtype=np.int)
        self.map_vector = np.zeros((self.mapping.shape[0],), dtype=np.int64)

        for source_label, target_label in self.mapping:
            self.map_vector[source_label] = target_label

    def get_metadata(self, name):
        img_file = self.root / 'leftImg8bit' / self.set_name / name
        label_name = name.replace("leftImg8bit", "gtFine_labelIds")
        label_file = self.root / 'gtFine' / self.set_name / label_name
        return img_file, label_file

    def map_labels(self, input_):
        return self.map_vector[input_.astype(np.int64, copy=False)]

    def __getitem__(self, index):
        img_file, label_file, _ = self.files[index]
        label = self.get_labels(label_file)
        label = self.map_labels(label).copy()
        image = self.get_image(img_file)
        image = self.preprocess(image)

        # for semi supervised setting
        if self.semi:
            semi_index = random.randint(0, len(self.semi_files)-1)
            semi_img_file, semi_label_file, _ = self.semi_files[semi_index]
            semi_label = self.get_labels(semi_label_file)
            semi_label = self.map_labels(semi_label).copy()
            semi_image = self.get_image(semi_img_file)
            semi_image = self.preprocess(semi_image)
            semi_image = semi_image.copy()
        else:
            semi_image, semi_label = [], []

        return image.copy(), label
