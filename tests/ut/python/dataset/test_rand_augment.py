# Copyright 2022 Huawei Technologies Co., Ltd
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
Testing RandAugment in DE
"""
import numpy as np

import mindspore.dataset as ds
from mindspore.dataset.vision import Decode, RandAugment, Resize
from mindspore.dataset.vision import Inter
from mindspore import log as logger
from util import visualize_image, visualize_list, diff_mse

IMAGE_FILE = "../data/dataset/testImageNetData/train/class1/1_1.jpg"
DATA_DIR = "../data/dataset/testImageNetData/train/"


def test_rand_augment_pipeline(plot=False):
    """
    Feature: RandAugment
    Description: Test RandAugment pipeline
    Expectation: Pass without error
    """
    logger.info("Test RandAugment pipeline")

    # Original Images
    data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
    transforms_original = [Decode(), Resize(size=[224, 224])]
    ds_original = data_set.map(operations=transforms_original, input_columns="image")
    ds_original = ds_original.batch(512)

    for idx, (image, _) in enumerate(ds_original):
        if idx == 0:
            images_original = image.asnumpy()
        else:
            images_original = np.append(images_original, image.asnumpy(), axis=0)

    data_set1 = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
    rand_augment_op = RandAugment(3, 10, 15, Inter.BICUBIC, 20)
    transforms = [Decode(), Resize(size=[224, 224]), rand_augment_op]
    ds_rand_augment = data_set1.map(operations=transforms, input_columns="image")
    ds_rand_augment = ds_rand_augment.batch(512)
    for idx, (image, _) in enumerate(ds_rand_augment):
        if idx == 0:
            images_rand_augment = image.asnumpy()
        else:
            images_rand_augment = np.append(images_rand_augment, image.asnumpy(), axis=0)
    assert images_original.shape[0] == images_rand_augment.shape[0]
    if plot:
        visualize_list(images_original, images_rand_augment)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_rand_augment[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))


def test_rand_augment_eager(plot=False):
    """
    Feature: RandAugment
    Description: Test RandAugment in eager mode
    Expectation: Pass without error
    """
    img = np.fromfile(IMAGE_FILE, dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    img = Decode()(img)
    img_rand_augmented = RandAugment()(img)
    if plot:
        visualize_image(img, img_rand_augmented)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img_rand_augmented), img_rand_augmented.shape))
    mse = diff_mse(img_rand_augmented, img)
    logger.info("MSE= {}".format(str(mse)))


def test_rand_augment_invalid_params_int():
    """
    Feature: RandAugment
    Description: Test RandAugment with invalid first three parameters
    Expectation: Error is raised as expected
    """
    logger.info("test_rand_augment_invalid_params_int")
    dataset = ds.ImageFolderDataset(DATA_DIR, 1, shuffle=False, decode=True)
    try:
        rand_augment_op = RandAugment(num_ops=-1)
        dataset.map(operations=rand_augment_op, input_columns=['image'])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input num_ops is not within the required interval of [0, 16777216]." in str(e)
    try:
        rand_augment_op = RandAugment(magnitude=-1)
        dataset.map(operations=rand_augment_op, input_columns=['image'])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input magnitude is not within the required interval of [0, 31)." in str(e)
    try:
        rand_augment_op = RandAugment(magnitude=0, num_magnitude_bins=1)
        dataset.map(operations=rand_augment_op, input_columns=['image'])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input num_magnitude_bins is not within the required interval of [2, 16777216]." in str(e)


def test_rand_augment_invalid_interpolation():
    """
    Feature: RandAugment
    Description: Test RandAugment with invalid interpolation
    Expectation: Error is raised as expected
    """
    logger.info("test_rand_augment_invalid_interpolation")
    dataset = ds.ImageFolderDataset(DATA_DIR, 1, shuffle=False, decode=True)
    try:
        rand_augment_op = RandAugment(interpolation="invalid")
        dataset.map(operations=rand_augment_op, input_columns=['image'])
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Argument interpolation with value invalid is not of type [<enum 'Inter'>]" in str(e)


def test_rand_augment_invalid_fill_value():
    """
    Feature: RandAugment
    Description: Test RandAugment with invalid fill_value
    Expectation: Correct error is raised as expected
    """
    logger.info("test_rand_augment_invalid_fill_value")
    dataset = ds.ImageFolderDataset(DATA_DIR, 1, shuffle=False, decode=True)
    try:
        rand_augment_op = RandAugment(fill_value=(10, 10))
        dataset.map(operations=rand_augment_op, input_columns=['image'])
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "fill_value should be a single integer or a 3-tuple." in str(e)
    try:
        rand_augment_op = RandAugment(fill_value=-1)
        dataset.map(operations=rand_augment_op, input_columns=['image'])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "is not within the required interval of [0, 255]." in str(e)


def test_rand_augment_invalid_magnitude_value():
    """
    Feature: RandAugment
    Description: Test RandAugment with invalid magnitude_value
    Expectation: Correct error is raised as expected
    """
    logger.info("test_rand_augment_invalid_magnitude_value")
    try:
        _ = RandAugment(3, 4, 3)
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input magnitude is not within the required interval of [0, 3)." in str(e)


if __name__ == "__main__":
    test_rand_augment_pipeline(plot=True)
    test_rand_augment_eager()
    test_rand_augment_invalid_params_int()
    test_rand_augment_invalid_interpolation()
    test_rand_augment_invalid_fill_value()
    test_rand_augment_invalid_magnitude_value()
