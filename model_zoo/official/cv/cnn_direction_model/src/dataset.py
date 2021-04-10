# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
Data operations, will be used in train.py and eval.py
"""
import os
import cv2
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
from src.dataset_utils import lucky, noise_blur, noise_speckle, noise_gamma, noise_gaussian, noise_salt_pepper, \
    shift_color, enhance_brightness, enhance_sharpness, enhance_contrast, enhance_color, gaussian_blur, \
    randcrop, resize, rdistort, rgeometry, rotate_about_center, whole_rdistort, warp_perspective, random_contrast, \
    unify_img_label

cv2.setNumThreads(0)

image_height = None
image_width = None


class Augmentor():
    """
     Augment image with random noise and transformation

     Controlled by severity level [0, 1]

     Usage:
         augmentor = Augmentor(severity=0.3,
                               prob=0.5,
                               enable_transform=True,
                               enable_crop=False)
         image_new = augmentor.process(image)
     """

    def __init__(self, severity, prob, enable_transform=True, enable_crop=False):
        """
        severity: in [0, 1], from min to max level of noise/transformation
        prob: in [0, 1], probability to apply each operator
        enable_transform: enable all transformation operators
        enable_crop: enable crop operator
        """
        self.severity = np.clip(severity, 0, 1)
        self.prob = np.clip(prob, 0, 1)
        self.enable_transform = enable_transform
        self.enable_crop = enable_crop

    def add_noise(self, im):
        """randomly add noise to image"""

        severity = self.severity
        prob = self.prob

        if lucky(prob):
            im = noise_gamma(im, severity=severity)
        if lucky(prob):
            im = noise_blur(im, severity=severity)
        if lucky(prob):
            im = noise_gaussian(im, severity=severity)
        if lucky(prob):
            im = noise_salt_pepper(im, severity=severity)
        if lucky(prob):
            im = shift_color(im, severity=severity)
        if lucky(prob):
            im = gaussian_blur(im, severity=severity)
        if lucky(prob):
            im = noise_speckle(im, severity=severity)
        if lucky(prob):
            im = enhance_sharpness(im, severity=severity)
        if lucky(prob):
            im = enhance_contrast(im, severity=severity)
        if lucky(prob):
            im = enhance_brightness(im, severity=severity)
        if lucky(prob):
            im = enhance_color(im, severity=severity)
        if lucky(prob):
            im = random_contrast(im)

        return im

    def convert_color(self, im, cval):
        if cval in ['median', 'md']:
            cval = np.median(im, axis=(0, 1)).astype(int)
        elif cval == 'mean':
            cval = np.mean(im, axis=(0, 1)).astype(int)
        if hasattr(cval, '__iter__'):
            cval = [int(i) for i in cval]
        else:
            cval = int(cval)
        return cval

    def transform(self, im, cval=255, **kw):
        """According to the parameters initialized by the class, deform the incoming image"""
        severity = self.severity
        prob = self.prob
        cval = self.convert_color(im, cval)
        if lucky(prob):
            # affine transform
            im = rgeometry(im, severity=severity, cval=cval)
        if lucky(prob):
            im = rdistort(im, severity=severity, cval=cval)
        if lucky(prob):
            im = warp_perspective(im, severity=severity, cval=cval)
        if lucky(prob):
            im = resize(im, fx=kw.get('fx'), fy=kw.get('fy'), severity=severity)
        if lucky(prob):
            im = rotate_about_center(im, severity=severity, cval=cval)
        if lucky(prob):
            # the overall distortion of the image.
            im = whole_rdistort(im, severity=severity)
        if lucky(prob) and self.enable_crop:
            # random crop
            im = randcrop(im, severity=severity)
        return im

    def process(self, im, cval='median', **kw):
        """ Execute code according to the effect of initial setting, and support variable parameters"""
        if self.enable_transform:
            im = self.transform(im, cval=cval, **kw)
        im = self.add_noise(im)
        return im


def rotate_and_set_neg(img, label):
    label = label - 1
    img_rotate = np.rot90(img)
    img_rotate = np.rot90(img_rotate)
    return img_rotate, np.array(label).astype(np.int32)

def crop_image(h_crop, w_crop):
    def crop_fun(img):
        return img[h_crop[0]:h_crop[1], w_crop[0]:w_crop[1], :]
    return crop_fun

def create_label(label=1):
    def label_fun(img):
        return img, np.array(label).astype(np.int32)
    return label_fun

def rotate(img, label):
    img_rotate = np.rot90(img)
    img_rotate = np.rot90(img_rotate)
    return img_rotate, label


def random_neg_with_rotate(img, label):
    if lucky(0.5):
        ##50% of samples set to  negative samples
        label = label - 1
        # rotate by 180 debgress
        img_rotate = np.rot90(img)
        img = np.rot90(img_rotate)
    return img, np.array(label).astype(np.int32)


def transform_image(img, label):
    data = np.array([img[...]], np.float32)
    data = data / 127.5 - 1
    return data.transpose((0, 3, 1, 2))[0], label


def create_dataset_train(mindrecord_file_pos, config, dataset_name='ocr'):
    """
    create a train dataset

    Args:
        mindrecord_file_pos(string): mindrecord file for positive samples.
        config(dict): config of dataset.
        dataset_name(string): name of dataset being used, e.g. 'fsns'.

    Returns:
        dataset
    """
    rank_size = int(os.getenv("RANK_SIZE", '1'))
    rank_id = int(os.getenv("RANK_ID", '0'))
    decode = C.Decode()
    columns_list = ["image", "label"] if dataset_name != 'fsns' else ["image"]
    data_set = ds.MindDataset(mindrecord_file_pos, columns_list=columns_list, num_parallel_workers=4,
                              num_shards=rank_size, shard_id=rank_id, shuffle=True)
    data_set = data_set.map(operations=decode, input_columns=["image"], num_parallel_workers=8)
    if dataset_name == 'fsns':
        data_set = data_set.map(operations=crop_image((0, 150), (0, 150)),
                                input_columns=["image"], num_parallel_workers=8)
        data_set = data_set.map(operations=create_label(), input_columns=["image"], output_columns=["image", "label"],
                                column_order=["image", "label"], num_parallel_workers=8)
    augmentor = Augmentor(config.augment_severity, config.augment_prob)
    operation = augmentor.process
    data_set = data_set.map(operations=operation, input_columns=["image"],
                            num_parallel_workers=1, python_multiprocessing=True)
    ##randomly augment half of samples to be negative samples
    data_set = data_set.map(operations=[random_neg_with_rotate, unify_img_label, transform_image],
                            input_columns=["image", "label"],
                            num_parallel_workers=8, python_multiprocessing=True)
    ##for training double the data_set to accoun for positive and negative
    data_set = data_set.repeat(2)

    # apply batch operations
    data_set = data_set.batch(config.batch_size, drop_remainder=True)
    return data_set


def resize_image(img, label):
    color_fill = 255
    scale = image_height / img.shape[0]
    img = cv2.resize(img, None, fx=scale, fy=scale)
    if img.shape[1] > image_width:
        img = img[:, 0:image_width]
    else:
        blank_img = np.zeros((image_height, image_width, 3), np.uint8)
        # fill the image with white
        blank_img.fill(color_fill)
        blank_img[:image_height, :img.shape[1]] = img
        img = blank_img
    data = np.array([img[...]], np.float32)
    data = data / 127.5 - 1
    return data.transpose((0, 3, 1, 2))[0], label


def create_dataset_eval(mindrecord_file_pos, config, dataset_name='ocr'):
    """
    create an eval dataset

    Args:
        mindrecord_file_pos(string): mindrecord file for positive samples.
        config(dict): config of dataset.

    Returns:
        dataset with images upright
        dataset with images 180-degrees rotated
    """
    rank_size = int(os.getenv("RANK_SIZE", '1'))
    rank_id = int(os.getenv("RANK_ID", '0'))
    decode = C.Decode()
    columns_list = ["image", "label"] if dataset_name != 'fsns' else ["image"]
    data_set = ds.MindDataset(mindrecord_file_pos, columns_list=columns_list, num_parallel_workers=1,
                              num_shards=rank_size, shard_id=rank_id, shuffle=False)
    data_set = data_set.map(operations=decode, input_columns=["image"], num_parallel_workers=8)
    if dataset_name == 'fsns':
        data_set = data_set.map(operations=crop_image((0, 150), (0, 150)),
                                input_columns=["image"], num_parallel_workers=8)
        data_set = data_set.map(operations=create_label(), input_columns=["image"], output_columns=["image", "label"],
                                column_order=["image", "label"], num_parallel_workers=8)
    global image_height
    global image_width
    image_height = config.im_size_h
    image_width = config.im_size_w
    data_set = data_set.map(operations=resize_image, input_columns=["image", "label"],
                            num_parallel_workers=config.work_nums,
                            python_multiprocessing=False)
    dataset_lr, dataset_rl = data_set.split([0.5, 0.5])
    dataset_rl = dataset_rl.map(operations=rotate_and_set_neg, input_columns=["image", "label"],
                                num_parallel_workers=config.work_nums,
                                python_multiprocessing=False)
    # apply batch operations
    dataset_lr = dataset_lr.batch(1, drop_remainder=True)
    dataset_rl = dataset_rl.batch(1, drop_remainder=True)

    return dataset_lr, dataset_rl
