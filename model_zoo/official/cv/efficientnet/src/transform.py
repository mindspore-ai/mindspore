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
random augment class
"""
import numpy as np
import mindspore.dataset.vision.py_transforms as P
from src import transform_utils


class RandAugment:
    # img_info have property： mean、std
    # config_str belongs to str
    # hparams belongs to dict
    def __init__(self, img_info, config_str="rand-m9-mstd0.5", hparams=None):
        hparams = hparams if hparams is not None else {}
        self.config_str = config_str
        self.hparams = hparams
        self.mean = img_info.mean
        self.std = img_info.std

    def __call__(self, imgs, labels, batchInfo):
        # assert the imgs object are pil_images
        ret_imgs = []
        ret_labels = []
        py_to_pil_op = P.ToPIL()
        to_tensor = P.ToTensor()
        normalize_op = P.Normalize(self.mean, self.std)
        rand_augment_ops = transform_utils.rand_augment_transform(self.config_str, self.hparams)
        for i, image in enumerate(imgs):
            img_pil = py_to_pil_op(image)
            img_pil = rand_augment_ops(img_pil)
            img_array = to_tensor(img_pil)
            img_array = normalize_op(img_array)
            ret_imgs.append(img_array)
            ret_labels.append(labels[i])
        return np.array(ret_imgs), np.array(ret_labels)
