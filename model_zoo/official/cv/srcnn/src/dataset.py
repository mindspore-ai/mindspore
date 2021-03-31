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

import glob
import numpy as np
import PIL.Image as pil_image

import mindspore.dataset as ds

from src.config import srcnn_cfg as config
from src.utils import convert_rgb_to_y

class EvalDataset:
    def __init__(self, images_dir):
        self.images_dir = images_dir
        scale = config.scale
        self.lr_group = []
        self.hr_group = []
        for image_path in sorted(glob.glob('{}/*'.format(images_dir))):
            hr = pil_image.open(image_path).convert('RGB')
            hr_width = (hr.width // scale) * scale
            hr_height = (hr.height // scale) * scale
            hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
            lr = hr.resize((hr_width // scale, hr_height // scale), resample=pil_image.BICUBIC)
            lr = lr.resize((lr.width * scale, lr.height * scale), resample=pil_image.BICUBIC)
            hr = np.array(hr).astype(np.float32)
            lr = np.array(lr).astype(np.float32)
            hr = convert_rgb_to_y(hr)
            lr = convert_rgb_to_y(lr)

            self.lr_group.append(lr)
            self.hr_group.append(hr)

    def __len__(self):
        return len(self.lr_group)

    def __getitem__(self, idx):
        return np.expand_dims(self.lr_group[idx] / 255., 0), np.expand_dims(self.hr_group[idx] / 255., 0)

def create_train_dataset(mindrecord_file, batch_size=1, shard_id=0, num_shard=1, num_parallel_workers=4):
    data_set = ds.MindDataset(mindrecord_file, columns_list=["lr", "hr"], num_shards=num_shard,
                              shard_id=shard_id, num_parallel_workers=num_parallel_workers, shuffle=True)
    data_set = data_set.batch(batch_size, drop_remainder=True)
    return data_set

def create_eval_dataset(images_dir, batch_size=1):
    dataset = EvalDataset(images_dir)
    data_set = ds.GeneratorDataset(dataset, ["lr", "hr"], shuffle=False)
    data_set = data_set.batch(batch_size, drop_remainder=True)
    return data_set
