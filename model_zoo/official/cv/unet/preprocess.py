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
"""unet 310 infer preprocess dataset"""
import os
import numpy as np
import cv2

from src.data_loader import create_dataset
from src.model_utils.config import config


def preprocess_dataset(data_dir, result_path, cross_valid_ind=1):

    _, valid_dataset = create_dataset(data_dir, 1, 1, False, cross_valid_ind, False, do_crop=config.crop,
                                      img_size=config.image_size)

    labels_list = []
    for i, data in enumerate(valid_dataset):
        file_name = "ISBI_test_bs_1_" + str(i) + ".bin"
        file_path = os.path.join(result_path, file_name)
        data[0].asnumpy().tofile(file_path)

        labels_list.append(data[1].asnumpy())

    np.save("./label.npy", labels_list)

class CellNucleiDataset:
    """
    Cell nuclei dataset preprocess class.
    """
    def __init__(self, data_dir, repeat, result_path, is_train=False, split=0.8):
        self.data_dir = data_dir
        self.img_ids = sorted(next(os.walk(self.data_dir))[1])
        self.train_ids = self.img_ids[:int(len(self.img_ids) * split)] * repeat
        self.val_ids = self.img_ids[int(len(self.img_ids) * split):]
        self.is_train = is_train
        self.result_path = result_path
        self._preprocess_dataset()

    def _preprocess_dataset(self):
        for img_id in self.val_ids:
            path = os.path.join(self.data_dir, img_id)
            img = cv2.imread(os.path.join(path, "images", img_id + ".png"))
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)
                img = np.concatenate([img, img, img], axis=-1)
            mask = []
            for mask_file in next(os.walk(os.path.join(path, "masks")))[2]:
                mask_ = cv2.imread(os.path.join(path, "masks", mask_file), cv2.IMREAD_GRAYSCALE)
                mask.append(mask_)
            mask = np.max(mask, axis=0)
            cv2.imwrite(os.path.join(self.result_path, img_id + ".png"), img)

    def _read_img_mask(self, img_id):
        path = os.path.join(self.data_dir, img_id)
        img = cv2.imread(os.path.join(path, "image.png"))
        mask = cv2.imread(os.path.join(path, "mask.png"), cv2.IMREAD_GRAYSCALE)
        return img, mask

    def __getitem__(self, index):
        if self.is_train:
            return self._read_img_mask(self.train_ids[index])
        return self._read_img_mask(self.val_ids[index])

    @property
    def column_names(self):
        column_names = ['image', 'mask']
        return column_names

    def __len__(self):
        if self.is_train:
            return len(self.train_ids)
        return len(self.val_ids)


if __name__ == '__main__':
    if hasattr(config, 'dataset') and config.dataset == "Cell_nuclei":
        cell_dataset = CellNucleiDataset(config.data_path, 1, config.result_path, False, 0.8)
    else:
        preprocess_dataset(data_dir=config.data_path, cross_valid_ind=config.cross_valid_ind,
                           result_path=config.result_path)
