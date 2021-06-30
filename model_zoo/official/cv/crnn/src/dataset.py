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
import numpy as np
from PIL import Image, ImageFile
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as vc
from src.model_utils.config import config as config1
from src.ic03_dataset import IC03Dataset
from src.ic13_dataset import IC13Dataset
from src.iiit5k_dataset import IIIT5KDataset
from src.svt_dataset import SVTDataset
ImageFile.LOAD_TRUNCATED_IMAGES = True


def check_image_is_valid(image):
    if image is None:
        return False

    h, w, c = image.shape
    if h * w * c == 0:
        return False

    return True

letters = [letter for letter in config1.label_dict]

def text_to_labels(text):
    return list(map(lambda x: letters.index(x.lower()), text))

class CaptchaDataset:
    """
    create train or evaluation dataset for crnn

    Args:
        img_root_dir(str): root path of images
        max_text_length(int): max number of digits in images.
        device_target(str): platform of training, support Ascend and GPU.
    """

    def __init__(self, img_root_dir, is_training=True, config=config1):
        if not os.path.exists(img_root_dir):
            raise RuntimeError("the input image dir {} is invalid!".format(img_root_dir))
        self.img_root_dir = img_root_dir
        if is_training:
            self.imgslist = os.path.join(self.img_root_dir, 'annotation_train.txt')
        else:
            self.imgslist = os.path.join(self.img_root_dir, 'annotation_test.txt')
        self.lexicon_file = os.path.join(self.img_root_dir, 'lexicon.txt')
        with open(self.lexicon_file, 'r') as f:
            self.lexicons = [line.strip('\n') for line in f]
        f.close()
        self.img_names = {}
        self.img_list = []
        with open(self.imgslist, 'r') as f:
            for line in f:
                img_name, label_index = line.strip('\n').split(" ")
                self.img_list.append(img_name)
                self.img_names[img_name] = self.lexicons[int(label_index)]
        f.close()
        self.max_text_length = config.max_text_length
        self.blank = config.blank
        self.class_num = config.class_num
        self.sample_num = len(self.img_names)
        self.batch_size = config.batch_size
        print("There are totally {} samples".format(self.sample_num))

    def __len__(self):
        return self.sample_num

    def __getitem__(self, item):
        img_name = self.img_list[item]
        try:
            im = Image.open(os.path.join(self.img_root_dir, img_name))
        except IOError:
            print("%s is a corrupted image" % img_name)
            return self[item + 1]
        im = im.convert("RGB")
        r, g, b = im.split()
        im = Image.merge("RGB", (b, g, r))
        image = np.array(im)
        if not check_image_is_valid(image):
            print("%s is a corrupted image" % img_name)
            return self[item + 1]

        text = self.img_names[img_name]
        label_unexpanded = text_to_labels(text)
        label = np.full(self.max_text_length, self.blank)
        if self.max_text_length < len(label_unexpanded):
            label_len = self.max_text_length
        else:
            label_len = len(label_unexpanded)
        for j in range(label_len):
            label[j] = label_unexpanded[j]
        return image, label


def create_dataset(name, dataset_path, batch_size=1, num_shards=1, shard_id=0, is_training=True, config=config1):
    """
     create train or evaluation dataset for crnn

     Args:
        dataset_path(int): dataset path
        batch_size(int): batch size of generated dataset, default is 1
        num_shards(int): number of devices
        shard_id(int): rank id
        device_target(str): platform of training, support Ascend and GPU
     """
    if name == 'synth':
        dataset = CaptchaDataset(dataset_path, is_training, config)
    elif name == 'ic03':
        dataset = IC03Dataset(dataset_path, "annotation.txt", config, True, 3)
    elif name == 'ic13':
        dataset = IC13Dataset(dataset_path, "Challenge2_Test_Task3_GT.txt", config)
    elif name == 'svt':
        dataset = SVTDataset(dataset_path, config)
    elif name == 'iiit5k':
        dataset = IIIT5KDataset(dataset_path, "annotation.txt", config)
    else:
        raise ValueError(f"unsupported dataset name: {name}")
    data_set = ds.GeneratorDataset(dataset, ["image", "label"], shuffle=True, num_shards=num_shards, shard_id=shard_id)
    image_trans = [
        vc.Resize((config.image_height, config.image_width)),
        vc.Normalize([127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
        vc.HWC2CHW()
    ]
    label_trans = [
        C.TypeCast(mstype.int32)
    ]
    data_set = data_set.map(operations=image_trans, input_columns=["image"], num_parallel_workers=8)
    data_set = data_set.map(operations=label_trans, input_columns=["label"], num_parallel_workers=8)

    data_set = data_set.batch(batch_size, drop_remainder=True)
    return data_set
