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
"""Dataset preprocessing."""
import os
import math as m
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.dataset.engine as de
import mindspore.dataset.transforms.c_transforms as c
import mindspore.dataset.transforms.vision.c_transforms as vc
from PIL import Image
from src.config import config as cf


class _CaptchaDataset():
    """
    create train or evaluation dataset for warpctc

    Args:
        img_root_dir(str): root path of images
        max_captcha_digits(int): max number of digits in images.
        blank(int): value reserved for blank label, default is 10. When parsing label from image file names, if label
        length is less than max_captcha_digits, the remaining labels are padding with blank.
    """

    def __init__(self, img_root_dir, max_captcha_digits, blank=10):
        if not os.path.exists(img_root_dir):
            raise RuntimeError("the input image dir {} is invalid!".format(img_root_dir))
        self.img_root_dir = img_root_dir
        self.img_names = [i for i in os.listdir(img_root_dir) if i.endswith('.png')]
        self.max_captcha_digits = max_captcha_digits
        self.blank = blank

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, item):
        img_name = self.img_names[item]
        im = Image.open(os.path.join(self.img_root_dir, img_name))
        r, g, b = im.split()
        im = Image.merge("RGB", (b, g, r))
        image = np.array(im)
        label_str = os.path.splitext(img_name)[0]
        label_str = label_str[label_str.find('-') + 1:]
        label = [int(i) for i in label_str]
        label.extend([int(self.blank)] * (self.max_captcha_digits - len(label)))
        label = np.array(label)
        return image, label


def create_dataset(dataset_path, repeat_num=1, batch_size=1):
    """
     create train or evaluation dataset for warpctc

     Args:
        dataset_path(int): dataset path
        repeat_num(int): dataset repetition num, default is 1
        batch_size(int): batch size of generated dataset, default is 1
     """
    rank_size = int(os.environ.get("RANK_SIZE")) if os.environ.get("RANK_SIZE") else 1
    rank_id = int(os.environ.get("RANK_ID")) if os.environ.get("RANK_ID") else 0

    dataset = _CaptchaDataset(dataset_path, cf.max_captcha_digits)
    ds = de.GeneratorDataset(dataset, ["image", "label"], shuffle=True, num_shards=rank_size, shard_id=rank_id)
    ds.set_dataset_size(m.ceil(len(dataset) / rank_size))
    image_trans = [
        vc.Rescale(1.0 / 255.0, 0.0),
        vc.Normalize([0.9010, 0.9049, 0.9025], std=[0.1521, 0.1347, 0.1458]),
        vc.Resize((m.ceil(cf.captcha_height / 16) * 16, cf.captcha_width)),
        vc.HWC2CHW()
    ]
    label_trans = [
        c.TypeCast(mstype.int32)
    ]
    ds = ds.map(input_columns=["image"], num_parallel_workers=8, operations=image_trans)
    ds = ds.map(input_columns=["label"], num_parallel_workers=8, operations=label_trans)

    ds = ds.batch(batch_size)
    ds = ds.repeat(repeat_num)
    return ds
