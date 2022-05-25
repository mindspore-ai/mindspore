# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""
Testing AutoAugment in DE
"""
import numpy as np

import mindspore.dataset as ds
from mindspore.dataset.vision import Decode, AutoAugment
from mindspore.dataset.vision.utils import AutoAugmentPolicy, Inter
from mindspore import log as logger
from util import helper_random_op_pipeline, visualize_image, visualize_list, diff_mse

image_file = "../data/dataset/testImageNetData/train/class1/1_1.jpg"
data_dir = "../data/dataset/testImageNetData/train/"


def test_auto_augment_pipeline(plot=False):
    """
    Feature: AutoAugment
    Description: Test AutoAugment pipeline
    Expectation: Pass without error
    """
    logger.info("Test AutoAugment pipeline")

    # Original images
    images_original = helper_random_op_pipeline(data_dir)

    # Auto Augmented Images with ImageNet policy
    auto_augment_op = AutoAugment(
        AutoAugmentPolicy.IMAGENET, Inter.BICUBIC, 20)
    images_auto_augment = helper_random_op_pipeline(
        data_dir, auto_augment_op)

    assert images_original.shape[0] == images_auto_augment.shape[0]
    if plot:
        visualize_list(images_original, images_auto_augment)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_auto_augment[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))

    # Auto Augmented Images with Cifar10 policy
    auto_augment_op = AutoAugment(
        AutoAugmentPolicy.CIFAR10, Inter.BILINEAR, 20)
    images_auto_augment = helper_random_op_pipeline(
        data_dir, auto_augment_op)
    assert images_original.shape[0] == images_auto_augment.shape[0]
    if plot:
        visualize_list(images_original, images_auto_augment)

    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_auto_augment[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))

    # Auto Augmented Images with SVHN policy
    auto_augment_op = AutoAugment(AutoAugmentPolicy.SVHN, Inter.NEAREST, 20)
    images_auto_augment = helper_random_op_pipeline(
        data_dir, auto_augment_op)
    assert images_original.shape[0] == images_auto_augment.shape[0]
    if plot:
        visualize_list(images_original, images_auto_augment)

    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_auto_augment[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))


def test_auto_augment_eager(plot=False):
    """
    Feature: AutoAugment
    Description: Test AutoAugment eager
    Expectation: Pass without error
    """
    img = np.fromfile(image_file, dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img = Decode()(img)
    img_auto_augmented = AutoAugment()(img)
    if plot:
        visualize_image(img, img_auto_augmented)
    logger.info("Image.type: {}, Image.shape: {}".format(
        type(img_auto_augmented), img_auto_augmented.shape))
    mse = diff_mse(img_auto_augmented, img)
    logger.info("MSE= {}".format(str(mse)))


def test_auto_augment_invalid_policy():
    """
    Feature: AutoAugment
    Description: Test AutoAugment with invalid policy
    Expectation: Throw correct error and message
    """
    logger.info("test_auto_augment_invalid_policy")
    dataset = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    try:
        auto_augment_op = AutoAugment(policy="invalid")
        dataset.map(operations=auto_augment_op, input_columns=['image'])
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Argument policy with value invalid is not of type [<enum 'AutoAugmentPolicy'>]" in str(
            e)


def test_auto_augment_invalid_interpolation():
    """
    Feature: AutoAugment
    Description: Test AutoAugment with invalid interpolation
    Expectation: Throw correct error and message
    """
    logger.info("test_auto_augment_invalid_interpolation")
    dataset = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    try:
        auto_augment_op = AutoAugment(interpolation="invalid")
        dataset.map(operations=auto_augment_op, input_columns=['image'])
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Argument interpolation with value invalid is not of type [<enum 'Inter'>]" in str(
            e)


def test_auto_augment_invalid_fill_value():
    """
    Feature: AutoAugment
    Description: Test AutoAugment with invalid fill_value
    Expectation: Throw correct error and message
    """
    logger.info("test_auto_augment_invalid_fill_value")
    dataset = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    try:
        auto_augment_op = AutoAugment(fill_value=(10, 10))
        dataset.map(operations=auto_augment_op, input_columns=['image'])
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "fill_value should be a single integer or a 3-tuple." in str(e)
    try:
        auto_augment_op = AutoAugment(fill_value=300)
        dataset.map(operations=auto_augment_op, input_columns=['image'])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "is not within the required interval of [0, 255]." in str(e)


if __name__ == "__main__":
    test_auto_augment_pipeline(plot=True)
    test_auto_augment_eager(plot=True)
    test_auto_augment_invalid_policy()
    test_auto_augment_invalid_interpolation()
    test_auto_augment_invalid_fill_value()
