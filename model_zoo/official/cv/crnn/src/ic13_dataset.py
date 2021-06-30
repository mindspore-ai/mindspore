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

"""Dataset adaptor for SVT"""
import os
import numpy as np
from PIL import Image, ImageFile
from src.model_utils.config import config as config1
ImageFile.LOAD_TRUNCATED_IMAGES = True


class IC13Dataset:
    """
    create evaluation dataset for crnn

    Args:
        img_root_dir(str): root path of images
        max_text_length(int): max number of digits in images
        device_target(str): platform of training, support Ascend and GPU
    """
    def __init__(self, img_root_dir, label_file="", config=config1, filter_by_dict=True, filter_length=3):
        if not os.path.exists(img_root_dir):
            raise RuntimeError("the input image dir {} is invalid".format(img_root_dir))
        self.img_root_dir = img_root_dir
        self.label_file = os.path.join(img_root_dir, label_file)
        self.img_names = {}
        self.img_list = []
        self.config = config
        with open(self.label_file, 'r') as f:
            for lines in f:
                img_name = lines.split(",")[0]
                label = lines.split("\"")[1].lower()
                if len(label) < filter_length:
                    continue
                if filter_by_dict:
                    flag = True
                    for c in label:
                        if c not in config.label_dict:
                            flag = False
                            break
                    if not flag:
                        continue
                self.img_names[img_name] = label
                self.img_list.append(img_name)
        f.close()
        self.max_text_length = config.max_text_length
        self.blank = config.blank
        self.class_num = config.class_num
        self.label_dict = config.label_dict
    def __len__(self):
        return len(self.img_names)
    def __getitem__(self, item):
        img_name = self.img_list[item]
        im = Image.open(os.path.join(self.img_root_dir, img_name))
        im = im.convert("RGB")
        r, g, b = im.split()
        im = Image.merge("RGB", (b, g, r))
        image = np.array(im)
        label_str = self.img_names[img_name]
        label = []
        for c in label_str:
            if c in self.label_dict:
                label.append(self.label_dict.index(c))
        label.extend([int(self.blank)] * (self.max_text_length - len(label)))
        label = np.array(label)
        return image, label
