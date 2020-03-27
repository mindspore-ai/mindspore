# Copyright 2019 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""Process imagenet validate dataset.
"""
import os

from mindspore import log as logger


def preprocess_imagenet_validation_dataset(train_dataset_path, validation_dataset_path, image_label_mapping_file):
    """
    Call this function before read imagenet validation dataset.

    Args:
        train_dataset_path (str): train dataset path
        validation_dataset_path (str): validation dataset path
        image_label_mapping_file (str): imagenet_validate_dataset_2012_image_dir_map.txt file path
    """
    train_dataset_path = os.path.realpath(train_dataset_path)
    sub_dir = [dir.name for dir in os.scandir(train_dataset_path) if dir.is_dir()]
    for sub_dir_name in sub_dir:
        validate_sub_dir = os.path.join(validation_dataset_path, sub_dir_name)
        validate_sub_dir = os.path.realpath(validate_sub_dir)
        if not os.path.exists(validate_sub_dir):
            os.makedirs(validate_sub_dir)

    mappings = [mapping.strip() for mapping in open(image_label_mapping_file).readlines()]
    for mapping in mappings:
        image_dir = mapping.split(':')
        old_image_path = os.path.join(validation_dataset_path, image_dir[0])
        old_image_path = os.path.realpath(old_image_path)
        if not os.path.exists(old_image_path):
            logger.warning('Image is not existed %s', old_image_path)
        new_image_sub_dir = os.path.join(validation_dataset_path, image_dir[1])
        new_image_sub_dir = os.path.realpath(new_image_sub_dir)
        new_image_path = os.path.join(new_image_sub_dir, image_dir[0])
        new_image_path = os.path.realpath(new_image_path)
        if not os.path.exists(new_image_sub_dir):
            logger.warning('Image sub dir is not existed %s', new_image_sub_dir)
        os.rename(old_image_path, new_image_path)
