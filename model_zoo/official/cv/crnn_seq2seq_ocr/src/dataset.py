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
"""FSNS dataset"""

import cv2
import numpy as np
from PIL import Image

import mindspore.dataset as de
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.vision.py_transforms as P
import mindspore.dataset.transforms.c_transforms as ops
import mindspore.common.dtype as mstype

from src.config import config


class AugmentationOps():
    def __init__(self, min_area_ratio=0.8, aspect_ratio_range=(0.8, 1.2), brightness=32./255.,
                 contrast=0.5, saturation=0.5, hue=0.2, img_tile_shape=(150, 150)):
        self.min_area_ratio = min_area_ratio
        self.aspect_ratio_range = aspect_ratio_range
        self.img_tile_shape = img_tile_shape
        self.random_image_distortion_ops = P.RandomColorAdjust(brightness=brightness,
                                                               contrast=contrast,
                                                               saturation=saturation,
                                                               hue=hue)

    def __call__(self, img):
        img_h = self.img_tile_shape[0]
        img_w = self.img_tile_shape[1]
        img_new = np.zeros([128, 512, 3])

        for i in range(4):
            img_tile = img[:, (i*150):((i+1)*150), :]
            # Random crop cut from the street sign image, resized to the same size.
            # Assures that the crop covers at least 0.8 area of the input image.
            # Aspect ratio of cropped image is within [0.8,1.2] range.
            h = img_h + 1
            w = img_w + 1

            while (w >= img_w or h >= img_h):
                aspect_ratio = np.random.uniform(self.aspect_ratio_range[0],
                                                 self.aspect_ratio_range[1])
                h_low = np.ceil(np.sqrt(self.min_area_ratio * img_h * img_w / aspect_ratio))
                h_high = np.floor(np.sqrt(img_h * img_w / aspect_ratio))
                h = np.random.randint(h_low, h_high)
                w = int(h * aspect_ratio)

            y = np.random.randint(img_w - w)
            x = np.random.randint(img_h - h)
            img_tile = img_tile[x:(x+h), y:(y+w), :]
            # Randomly chooses one of the 4 interpolation resize methods.
            interpolation = np.random.choice([cv2.INTER_LINEAR,
                                              cv2.INTER_CUBIC,
                                              cv2.INTER_AREA,
                                              cv2.INTER_NEAREST])
            img_tile = cv2.resize(img_tile, (128, 128), interpolation=interpolation)
            # Random color distortion ops.
            img_tile_pil = Image.fromarray(img_tile)
            img_tile_pil = self.random_image_distortion_ops(img_tile_pil)
            img_tile = np.array(img_tile_pil)
            img_new[:, (i*128):((i+1)*128), :] = img_tile

        img_new = 2 * (img_new / 255.) - 1
        return img_new


class ImageResizeWithRescale():
    def __init__(self, standard_img_height, standard_img_width, channel_size=3):
        self.standard_img_height = standard_img_height
        self.standard_img_width = standard_img_width
        self.channel_size = channel_size

    def __call__(self, img):
        img = cv2.resize(img, (self.standard_img_width, self.standard_img_height))
        img = 2 * (img / 255.) - 1
        return img


def random_teacher_force(images, source_ids, target_ids):
    teacher_force = np.random.random() < config.teacher_force_ratio
    teacher_force_array = np.array([teacher_force], dtype=bool)
    return images, source_ids, target_ids, teacher_force_array


def create_ocr_train_dataset(mindrecord_file, batch_size=32, rank_size=1, rank_id=0,
                             is_training=True, num_parallel_workers=4, use_multiprocessing=True):
    ds = de.MindDataset(mindrecord_file,
                        columns_list=["image", "decoder_input", "decoder_target"],
                        num_shards=rank_size,
                        shard_id=rank_id,
                        num_parallel_workers=num_parallel_workers,
                        shuffle=is_training)
    aug_ops = AugmentationOps()
    transforms = [C.Decode(),
                  aug_ops,
                  C.HWC2CHW()]
    ds = ds.map(operations=transforms, input_columns=["image"], python_multiprocessing=use_multiprocessing,
                num_parallel_workers=num_parallel_workers)
    ds = ds.map(operations=ops.PadEnd([config.max_length], 0), input_columns=["decoder_target"])
    ds = ds.map(operations=random_teacher_force, input_columns=["image", "decoder_input", "decoder_target"],
                output_columns=["image", "decoder_input", "decoder_target", "teacher_force"],
                column_order=["image", "decoder_input", "decoder_target", "teacher_force"])
    type_cast_op_bool = ops.TypeCast(mstype.bool_)
    ds = ds.map(operations=type_cast_op_bool, input_columns="teacher_force")
    print("Train dataset size= %s" % (int(ds.get_dataset_size())))
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds


def create_ocr_val_dataset(mindrecord_file, batch_size=32, rank_size=1, rank_id=0,
                           num_parallel_workers=4, use_multiprocessing=True):
    ds = de.MindDataset(mindrecord_file,
                        columns_list=["image", "annotation", "decoder_input", "decoder_target"],
                        num_shards=rank_size,
                        shard_id=rank_id,
                        num_parallel_workers=num_parallel_workers,
                        shuffle=False)
    resize_rescale_op = ImageResizeWithRescale(standard_img_height=128, standard_img_width=512)
    transforms = [C.Decode(),
                  resize_rescale_op,
                  C.HWC2CHW()]
    ds = ds.map(operations=transforms, input_columns=["image"], python_multiprocessing=use_multiprocessing,
                num_parallel_workers=num_parallel_workers)
    ds = ds.map(operations=ops.PadEnd([config.max_length], 0), input_columns=["decoder_target"],
                python_multiprocessing=use_multiprocessing, num_parallel_workers=8)
    ds = ds.map(operations=ops.PadEnd([config.max_length], 0), input_columns=["decoder_input"],
                python_multiprocessing=use_multiprocessing, num_parallel_workers=8)
    ds = ds.batch(batch_size, drop_remainder=True)
    print("Val dataset size= %s" % (str(int(ds.get_dataset_size())*batch_size)))
    return ds
