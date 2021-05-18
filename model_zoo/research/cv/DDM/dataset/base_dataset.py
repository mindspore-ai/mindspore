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

"""basic dataset setting."""

from pathlib import Path
import random
import numpy as np
from PIL import Image
from skimage import color

TARGET_IMGS = None

class BaseDataset():
    """basic dataset setting"""
    def __init__(self, root, list_path, set_, max_iters, image_size,
                 labels_size, mean, semi=False, num_semi=100, trans_img=False, del_xl=False):
        self.root = Path(root)
        self.set_name = set_
        self.list_path = list_path.format(self.set_name)

        if labels_size is None:
            self.labels_size = self.image_size
        else:
            self.labels_size = labels_size

        self.mean = mean

        with open(self.list_path) as f:
            self.img_ids = [i_id.strip() for i_id in f]

        repeat_num, repeat_num_semi = 1, 1
        self.files = []

        for name in self.img_ids:
            img_file, label_file = self.get_metadata(name)
            self.files.append((img_file, label_file, name))

        if semi:
            print("Semi-supervised setting is used, the number of images " \
                  "for supervised training is {}".format(num_semi))
            self.semi_files_sel = random.sample(self.files, num_semi)
            if del_xl:
                for i in self.semi_files_sel:
                    self.files.remove(i)
            if max_iters is not None:
                repeat_num_semi = int(np.ceil(float(max_iters) / num_semi))
            self.semi_files = self.semi_files_sel * repeat_num_semi
        else:
            self.semi_files = None

        if max_iters is not None:
            repeat_num = int(np.ceil(float(max_iters) / len(self.files)))

        self.files = self.files * repeat_num

        # for trans_img setting
        self.trans_img = trans_img

        if trans_img:
            global TARGET_IMGS
            print('Get the target images list for data trans')
            with open('dataset/cityscapes_list/train.txt', 'r') as f:
                TARGET_IMGS = [f'{root}/../cityscapes/leftImg8bit/train/'+x.strip() for x in f]


    def get_metadata(self, name):
        raise NotImplementedError

    def __len__(self):
        return len(self.files)

    def preprocess(self, image):
        # change to BGR
        image = image[:, :, ::-1]
        image -= self.mean
        return image.transpose((2, 0, 1))

    def get_image(self, file):
        return _load_img(file, self.image_size, Image.BICUBIC, rgb=True, trans_img=self.trans_img)

    def get_labels(self, file):
        return _load_img(file, self.labels_size, Image.NEAREST, rgb=False, trans_img=False)


def _load_img(file, size, interpolation, rgb, trans_img=False):
    """load images"""
    img = Image.open(file)

    if rgb:
        # translate to the target style
        if trans_img:
            img = np.array(img)
            t_img = np.array(Image.open(random.choices(TARGET_IMGS)[0]))
            lab = color.rgb2lab(img)
            t_lab = color.rgb2lab(t_img)
            for i in range(3):
                lab[:, :, i] = (lab[:, :, i] - lab[:, :, i].mean()) / lab[:, :, i].std() * t_lab[:, :, i].std()\
                                + t_lab[:, :, i].mean()
            img = color.lab2rgb(lab) * 255
            img = np.clip(img, 0, 255)
            img = Image.fromarray(img.astype(np.uint8))
        # end
        img = img.convert('RGB')

    img = img.resize(size, interpolation)
    return np.asarray(img, np.float32)
