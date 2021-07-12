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

"""SRGAN test dataset."""

import os
import numpy as np
from PIL import Image
import mindspore.dataset as ds


class mydata:
    """Import dataset"""
    def __init__(self, LR_path, GT_path, in_memory=True):
        self.LR_path = LR_path
        self.GT_path = GT_path
        self.in_memory = in_memory
        self.LR_img = sorted(os.listdir(LR_path))
        self.GT_img = sorted(os.listdir(GT_path))
        if in_memory:
            self.LR_img = [np.array(Image.open(os.path.join(self.LR_path, LR)).convert("RGB")).astype(np.float32)
                           for LR in self.LR_img]
            self.GT_img = [np.array(Image.open(os.path.join(self.GT_path, HR)).convert("RGB")).astype(np.float32)
                           for HR in self.GT_img]

    def __len__(self):
        """length"""
        return len(self.LR_img)

    def __getitem__(self, i):
        """getitem"""
        img_item = {}
        if self.in_memory:
            GT = self.GT_img[i].astype(np.float32)
            LR = self.LR_img[i].astype(np.float32)

        else:
            GT = np.array(Image.open(os.path.join(self.GT_path, self.GT_img[i])).convert("RGB"))
            LR = np.array(Image.open(os.path.join(self.LR_path, self.LR_img[i])).convert("RGB"))
        img_item['GT'] = (GT / 127.5) - 1.0
        img_item['LR'] = (LR / 127.5) - 1.0
        img_item['GT'] = img_item['GT'].transpose(2, 0, 1).astype(np.float32)
        img_item['LR'] = img_item['LR'].transpose(2, 0, 1).astype(np.float32)

        return  img_item['LR'], img_item['GT']

def create_testdataset(batchsize, LR_path, GT_path):
    """create testdataset"""
    dataset = mydata(LR_path, GT_path, in_memory=False)
    DS = ds.GeneratorDataset(dataset, column_names=["LR", "HR"])
    DS = DS.batch(batchsize)
    return DS
