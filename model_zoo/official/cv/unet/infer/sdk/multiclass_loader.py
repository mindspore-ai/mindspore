# coding=utf-8
#
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

import os

import cv2 as cv


class MultiClassLoader:
    _plugin_name = "multiclass"

    def __init__(self, dataset_dir):
        super(MultiClassLoader, self).__init__()
        self._dataset_dir = dataset_dir

    def iter_dataset(self):
        for image_id in self.list_image_id():
            image, mask = self.get_image_mask(image_id)
            yield image_id, image, mask

    def list_image_id(self):
        return os.listdir(self._dataset_dir)

    def get_image_mask(self, image_id):
        image_dir = os.path.join(self._dataset_dir, image_id)
        image = cv.imread(os.path.join(image_dir, "image.png"))
        mask = cv.imread(os.path.join(image_dir, "mask.png"), cv.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise RuntimeError(f"Failed to get image by id {image_id}")

        return image, mask

    @classmethod
    def get_plugin_name(cls):
        return cls._plugin_name
