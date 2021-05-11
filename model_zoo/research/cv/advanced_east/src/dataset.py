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
"""
dataset.
"""
import os

import mindspore.dataset as de
import mindspore.dataset.vision.c_transforms as vision
from mindspore.mindrecord import FileWriter
import numpy as np
from PIL import Image, ImageFile

import src.config as cfg

ImageFile.LOAD_TRUNCATED_IMAGES = True
cfg = cfg.config


def gen(batch_size=cfg.batch_size, is_val=False):
    """generate label"""
    img_h, img_w = cfg.max_train_img_size, cfg.max_train_img_size
    x = np.zeros((batch_size, 3, img_h, img_w), dtype=np.float32)
    pixel_num_h = img_h // 4
    pixel_num_w = img_w // 4
    y = np.zeros((batch_size, 7, pixel_num_h, pixel_num_w), dtype=np.float32)
    if is_val:
        with open(os.path.join(cfg.data_dir, cfg.val_fname), 'r') as f_val:
            f_list = f_val.readlines()
    else:
        with open(os.path.join(cfg.data_dir, cfg.train_fname), 'r') as f_train:
            f_list = f_train.readlines()
    while True:
        batch_list = np.arange(0, len(f_list), batch_size)
        for idx in batch_list:
            for idx2 in range(idx, idx + batch_size):
                # random gen an image name
                if idx2 < len(f_list):
                    img = f_list[idx2]
                else:
                    img = np.random.choice(f_list)
                img_filename = str(img).strip().split(',')[0]
                # load img and img anno
                img_path = os.path.join(cfg.data_dir,
                                        cfg.train_image_dir_name,
                                        img_filename)
                img = Image.open(img_path)
                img = np.asarray(img)
                img = img / 127.5 - 1
                x[idx2 - idx] = img.transpose((2, 0, 1))
                gt_file = os.path.join(cfg.data_dir,
                                       cfg.train_label_dir_name,
                                       img_filename[:-4] + '_gt.npy')
                y[idx2 - idx] = np.load(gt_file).transpose((2, 0, 1))
            yield x, y


def transImage2Mind(mindrecord_filename, is_val=False):
    """transfer the image to mindrecord"""
    if os.path.exists(mindrecord_filename):
        os.remove(mindrecord_filename)
        os.remove(mindrecord_filename + ".db")

    writer = FileWriter(file_name=mindrecord_filename, shard_num=1)
    cv_schema = {"image": {"type": "bytes"}, "label": {"type": "float32", "shape": [7, 112, 112]}}
    writer.add_schema(cv_schema, "advancedEast dataset")

    if is_val:
        with open(os.path.join(cfg.data_dir, cfg.val_fname), 'r') as f_val:
            f_list = f_val.readlines()
    else:
        with open(os.path.join(cfg.data_dir, cfg.train_fname), 'r') as f_train:
            f_list = f_train.readlines()

    data = []
    for item in f_list:
        img_filename = str(item).strip().split(',')[0]
        img_path = os.path.join(cfg.data_dir,
                                cfg.train_image_dir_name,
                                img_filename)
        print(img_path)
        with open(img_path, 'rb') as f:
            img = f.read()

        gt_file = os.path.join(cfg.data_dir,
                               cfg.train_label_dir_name,
                               img_filename[:-4] + '_gt.npy')
        labels = np.load(gt_file)
        labels = np.transpose(labels, (2, 0, 1))
        data.append({"image": img,
                     "label": np.array(labels, np.float32)})
        if len(data) == 32:
            writer.write_raw_data(data)
            print('len(data):{}'.format(len(data)))
            data = []
    if data:
        writer.write_raw_data(data)
    writer.commit()


def transImage2Mind_size(mindrecord_filename, width=256, is_val=False):
    """transfer the image to mindrecord at specified size"""
    mindrecord_filename = mindrecord_filename + str(width) + '.mindrecord'
    if os.path.exists(mindrecord_filename):
        os.remove(mindrecord_filename)
        os.remove(mindrecord_filename + ".db")

    writer = FileWriter(file_name=mindrecord_filename, shard_num=1)
    cv_schema = {"image": {"type": "bytes"}, "label": {"type": "float32", "shape": [7, width // 4, width // 4]}}
    writer.add_schema(cv_schema, "advancedEast dataset")

    if is_val:
        with open(os.path.join(cfg.data_dir, cfg.val_fname_var + str(width) + '.txt'), 'r') as f_val:
            f_list = f_val.readlines()
    else:
        with open(os.path.join(cfg.data_dir, cfg.train_fname_var + str(width) + '.txt'), 'r') as f_train:
            f_list = f_train.readlines()
    data = []
    for item in f_list:

        img_filename = str(item).strip().split(',')[0]
        img_path = os.path.join(cfg.data_dir,
                                cfg.train_image_dir_name_var + str(width),
                                img_filename)
        with open(img_path, 'rb') as f:
            img = f.read()
        gt_file = os.path.join(cfg.data_dir,
                               cfg.train_label_dir_name_var + str(width),
                               img_filename[:-4] + '_gt.npy')
        labels = np.load(gt_file)
        labels = np.transpose(labels, (2, 0, 1))
        data.append({"image": img,
                     "label": np.array(labels, np.float32)})

        if len(data) == 32:
            writer.write_raw_data(data)
            print('len(data):{}'.format(len(data)))
            data = []

    if data:
        writer.write_raw_data(data)
    writer.commit()


def load_adEAST_dataset(mindrecord_file, batch_size=64, device_num=1, rank_id=0,
                        is_training=True, num_parallel_workers=8):
    """load mindrecord"""
    ds = de.MindDataset(mindrecord_file, columns_list=["image", "label"], num_shards=device_num, shard_id=rank_id,
                        num_parallel_workers=8, shuffle=is_training)
    hwc_to_chw = vision.HWC2CHW()
    cd = vision.Decode()
    ds = ds.map(operations=cd, input_columns=["image"])
    rc = vision.RandomColorAdjust(brightness=0.1, contrast=0.2, saturation=0.2)
    vn = vision.Normalize(mean=(123.68, 116.779, 103.939), std=(1., 1., 1.))
    ds = ds.map(operations=[rc, vn, hwc_to_chw], input_columns=["image"], num_parallel_workers=num_parallel_workers)
    ds = ds.batch(batch_size, drop_remainder=True)
    batch_num = ds.get_dataset_size()
    ds = ds.shuffle(batch_num)
    return ds, batch_num
