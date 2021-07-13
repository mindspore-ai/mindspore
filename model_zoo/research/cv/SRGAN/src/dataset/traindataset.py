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

"""SRGAN train dataset."""

import os
import random
import math
import numpy as np
from PIL import Image
import mindspore.dataset as ds
from mindspore import context
from mindspore.context import ParallelMode

class mydata:
    """Import dataset"""
    def __init__(self, LR_path, GT_path, in_memory=True):
        """init"""
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
        """getlength"""
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
        # crop
        ih, iw = img_item['LR'].shape[:2]
        ix = random.randrange(0, iw -24 + 1)
        iy = random.randrange(0, ih -24 + 1)
        tx = ix * 4
        ty = iy * 4
        img_item['LR'] = img_item['LR'][iy : iy + 24, ix : ix + 24]
        img_item['GT'] = img_item['GT'][ty : ty + (4 * 24), tx : tx + (4 * 24)]
        # augmentation
        hor_flip = random.randrange(0, 2)
        ver_flip = random.randrange(0, 2)
        rot = random.randrange(0, 2)
        if hor_flip:
            temp_LR = np.fliplr(img_item['LR'])
            img_item['LR'] = temp_LR.copy()
            temp_GT = np.fliplr(img_item['GT'])
            img_item['GT'] = temp_GT.copy()
            del temp_LR, temp_GT

        if ver_flip:
            temp_LR = np.flipud(img_item['LR'])
            img_item['LR'] = temp_LR.copy()
            temp_GT = np.flipud(img_item['GT'])
            img_item['GT'] = temp_GT.copy()
            del temp_LR, temp_GT

        if rot:
            img_item['LR'] = img_item['LR'].transpose(1, 0, 2)
            img_item['GT'] = img_item['GT'].transpose(1, 0, 2)
        img_item['GT'] = img_item['GT'].transpose(2, 0, 1).astype(np.float32)
        img_item['LR'] = img_item['LR'].transpose(2, 0, 1).astype(np.float32)
        return  img_item['LR'], img_item['GT']

class MySampler():
    """sampler for distribution"""
    def __init__(self, dataset, local_rank, world_size):
        self.__num_data = len(dataset)
        self.__local_rank = local_rank
        self.__world_size = world_size
        self.samples_per_rank = int(math.ceil(self.__num_data / float(self.__world_size)))
        self.total_num_samples = self.samples_per_rank * self.__world_size

    def __iter__(self):
        """"iter"""
        indices = list(range(self.__num_data))
        indices.extend(indices[:self.total_num_samples-len(indices)])
        indices = indices[self.__local_rank:self.total_num_samples:self.__world_size]
        return iter(indices)

    def __len__(self):
        """length"""
        return self.samples_per_rank

def create_traindataset(batchsize, LR_path, GT_path):
    """"create SRGAN dataset"""
    parallel_mode = context.get_auto_parallel_context("parallel_mode")
    if parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
        dataset = mydata(LR_path, GT_path, in_memory=True)
        sampler = MySampler(dataset, local_rank=0, world_size=4)
        device_num = int(os.getenv("RANK_SIZE"))
        rank_id = int(os.getenv("DEVICE_ID"))
        sampler = MySampler(dataset, local_rank=rank_id, world_size=4)
        DS = ds.GeneratorDataset(dataset, column_names=['LR', 'HR'], shuffle=True,
                                 num_shards=device_num, shard_id=rank_id, sampler=sampler)
        DS = DS.batch(batchsize, drop_remainder=True)
    else:
        dataset = mydata(LR_path, GT_path, in_memory=True)
        DS = ds.GeneratorDataset(dataset, column_names=['LR', 'HR'], shuffle=True)
        DS = DS.batch(batchsize)
    return DS
