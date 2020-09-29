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
"""cnn_ctc dataset"""

import sys
import pickle
import math
import six
import numpy as np
from PIL import Image
import lmdb

from mindspore.communication.management import get_rank, get_group_size

from .util import CTCLabelConverter
from .config import Config_CNNCTC

config = Config_CNNCTC()

class NormalizePAD():

    def __init__(self, max_size, PAD_type='right'):
        self.max_size = max_size
        self.PAD_type = PAD_type

    def __call__(self, img):
        # toTensor
        img = np.array(img, dtype=np.float32)
        img = img.transpose([2, 0, 1])
        img = img.astype(np.float)
        img = np.true_divide(img, 255)
        # normalize
        img = np.subtract(img, 0.5)
        img = np.true_divide(img, 0.5)

        _, _, w = img.shape
        Pad_img = np.zeros(shape=self.max_size, dtype=np.float32)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = np.tile(np.expand_dims(img[:, :, w - 1], 2), (1, 1, self.max_size[2] - w))

        return Pad_img


class AlignCollate():

    def __init__(self, imgH=32, imgW=100):
        self.imgH = imgH
        self.imgW = imgW

    def __call__(self, images):

        resized_max_w = self.imgW
        input_channel = 3
        transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

        resized_images = []
        for image in images:
            w, h = image.size
            ratio = w / float(h)
            if math.ceil(self.imgH * ratio) > self.imgW:
                resized_w = self.imgW
            else:
                resized_w = math.ceil(self.imgH * ratio)

            resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
            resized_images.append(transform(resized_image))

        image_tensors = np.concatenate([np.expand_dims(t, 0) for t in resized_images], 0)

        return image_tensors


def get_img_from_lmdb(env, index):
    with env.begin(write=False) as txn:
        label_key = 'label-%09d'.encode() % index
        label = txn.get(label_key).decode('utf-8')
        img_key = 'image-%09d'.encode() % index
        imgbuf = txn.get(img_key)

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        try:
            img = Image.open(buf).convert('RGB')  # for color image

        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            img = Image.new('RGB', (config.IMG_W, config.IMG_H))
            label = '[dummy_label]'

    label = label.lower()

    return img, label


class ST_MJ_Generator_batch_fixed_length:
    def __init__(self):
        self.align_collector = AlignCollate()
        self.converter = CTCLabelConverter(config.CHARACTER)
        self.env = lmdb.open(config.TRAIN_DATASET_PATH, max_readers=32, readonly=True, lock=False, readahead=False,
                             meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (config.TRAIN_DATASET_PATH))
            raise ValueError(config.TRAIN_DATASET_PATH)

        with open(config.TRAIN_DATASET_INDEX_PATH, 'rb') as f:
            self.st_mj_filtered_index_list = pickle.load(f)

        print(f'num of samples in ST_MJ dataset: {len(self.st_mj_filtered_index_list)}')
        self.dataset_size = len(self.st_mj_filtered_index_list) // config.TRAIN_BATCH_SIZE
        self.batch_size = config.TRAIN_BATCH_SIZE

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item):
        img_ret = []
        text_ret = []

        for i in range(item * self.batch_size, (item + 1) * self.batch_size):
            index = self.st_mj_filtered_index_list[i]
            img, label = get_img_from_lmdb(self.env, index)

            img_ret.append(img)
            text_ret.append(label)

        img_ret = self.align_collector(img_ret)
        text_ret, length = self.converter.encode(text_ret)

        label_indices = []
        for i, _ in enumerate(length):
            for j in range(length[i]):
                label_indices.append((i, j))
        label_indices = np.array(label_indices, np.int64)
        sequence_length = np.array([config.FINAL_FEATURE_WIDTH] * config.TRAIN_BATCH_SIZE, dtype=np.int32)
        text_ret = text_ret.astype(np.int32)

        return img_ret, label_indices, text_ret, sequence_length

class ST_MJ_Generator_batch_fixed_length_para:
    def __init__(self):
        self.align_collector = AlignCollate()
        self.converter = CTCLabelConverter(config.CHARACTER)
        self.env = lmdb.open(config.TRAIN_DATASET_PATH, max_readers=32, readonly=True, lock=False, readahead=False,
                             meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (config.TRAIN_DATASET_PATH))
            raise ValueError(config.TRAIN_DATASET_PATH)

        with open(config.TRAIN_DATASET_INDEX_PATH, 'rb') as f:
            self.st_mj_filtered_index_list = pickle.load(f)

        print(f'num of samples in ST_MJ dataset: {len(self.st_mj_filtered_index_list)}')
        self.rank_id = get_rank()
        self.rank_size = get_group_size()
        self.dataset_size = len(self.st_mj_filtered_index_list) // config.TRAIN_BATCH_SIZE // self.rank_size
        self.batch_size = config.TRAIN_BATCH_SIZE

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item):
        img_ret = []
        text_ret = []

        rank_item = (item * self.rank_size) + self.rank_id
        for i in range(rank_item * self.batch_size, (rank_item + 1) * self.batch_size):
            index = self.st_mj_filtered_index_list[i]
            img, label = get_img_from_lmdb(self.env, index)

            img_ret.append(img)
            text_ret.append(label)

        img_ret = self.align_collector(img_ret)
        text_ret, length = self.converter.encode(text_ret)

        label_indices = []
        for i, _ in enumerate(length):
            for j in range(length[i]):
                label_indices.append((i, j))
        label_indices = np.array(label_indices, np.int64)
        sequence_length = np.array([config.FINAL_FEATURE_WIDTH] * config.TRAIN_BATCH_SIZE, dtype=np.int32)
        text_ret = text_ret.astype(np.int32)

        return img_ret, label_indices, text_ret, sequence_length


def IIIT_Generator_batch():
    max_len = int((26 + 1) // 2)

    align_collector = AlignCollate()

    converter = CTCLabelConverter(config.CHARACTER)

    env = lmdb.open(config.TEST_DATASET_PATH, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    if not env:
        print('cannot create lmdb from %s' % (config.TEST_DATASET_PATH))
        sys.exit(0)

    with env.begin(write=False) as txn:
        nSamples = int(txn.get('num-samples'.encode()))
        nSamples = nSamples

        # Filtering
        filtered_index_list = []
        for index in range(nSamples):
            index += 1  # lmdb starts with 1
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')

            if len(label) > max_len:
                continue

            illegal_sample = False
            for char_item in label.lower():
                if char_item not in config.CHARACTER:
                    illegal_sample = True
                    break
            if illegal_sample:
                continue

            filtered_index_list.append(index)

    img_ret = []
    text_ret = []

    print(f'num of samples in IIIT dataset: {len(filtered_index_list)}')

    for index in filtered_index_list:

        img, label = get_img_from_lmdb(env, index)

        img_ret.append(img)
        text_ret.append(label)

        if len(img_ret) == config.TEST_BATCH_SIZE:
            img_ret = align_collector(img_ret)
            text_ret, length = converter.encode(text_ret)

            label_indices = []
            for i, _ in enumerate(length):
                for j in range(length[i]):
                    label_indices.append((i, j))
            label_indices = np.array(label_indices, np.int64)
            sequence_length = np.array([26] * config.TEST_BATCH_SIZE, dtype=np.int32)
            text_ret = text_ret.astype(np.int32)

            yield img_ret, label_indices, text_ret, sequence_length, length

            img_ret = []
            text_ret = []
