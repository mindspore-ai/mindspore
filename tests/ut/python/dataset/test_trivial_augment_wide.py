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
Testing TrivialAugmentWide in DE
"""
import numpy as np

import mindspore.dataset as ds
from mindspore.dataset.vision.transforms import Decode, TrivialAugmentWide, Resize
from mindspore.dataset.vision.utils import Inter
from mindspore import log as logger
from util import visualize_image, visualize_list, diff_mse


def test_trivial_augment_wide_pipeline(plot=False):
    """
    Feature: TrivialAugmentWide
    Description: test TrivialAugmentWide pipeline
    Expectation: pass without error
    """
    logger.info("Test TrivialAugmentWide pipeline")
    data_dir = "../data/dataset/testImageNetData/train/"

    # Original Images
    data_set = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    transforms_original = [Decode(), Resize(size=[224, 224])]
    ds_original = data_set.map(operations=transforms_original, input_columns="image")
    ds_original = ds_original.batch(512)

    for idx, (image, _) in enumerate(ds_original):
        if idx == 0:
            images_original = image.asnumpy()
        else:
            images_original = np.append(images_original,
                                        image.asnumpy(),
                                        axis=0)

    # Trivial Augment Wided Images with ImageNet num_magnitude_bins
    data_set1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    trivial_augment_wide_op = TrivialAugmentWide(31, Inter.BICUBIC, 20)
    transforms = [Decode(), Resize(size=[224, 224]), trivial_augment_wide_op]
    ds_trivial_augment_wide = data_set1.map(operations=transforms, input_columns="image")
    ds_trivial_augment_wide = ds_trivial_augment_wide.batch(512)

    for idx, (image, _) in enumerate(ds_trivial_augment_wide):
        if idx == 0:
            images_trivial_augment_wide = image.asnumpy()
        else:
            images_trivial_augment_wide = np.append(images_trivial_augment_wide,
                                                    image.asnumpy(), axis=0)
    assert images_original.shape[0] == images_trivial_augment_wide.shape[0]
    if plot:
        visualize_list(images_original, images_trivial_augment_wide)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_trivial_augment_wide[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))

    # Trivial Augment Wided Images with Cifar10 num_magnitude_bins
    data_set2 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    trivial_augment_wide_op = TrivialAugmentWide(31, Inter.BILINEAR, 20)
    transforms = [Decode(), Resize(size=[224, 224]), trivial_augment_wide_op]
    ds_trivial_augment_wide = data_set2.map(operations=transforms, input_columns="image")
    ds_trivial_augment_wide = ds_trivial_augment_wide.batch(512)
    for idx, (image, _) in enumerate(ds_trivial_augment_wide):
        if idx == 0:
            images_trivial_augment_wide = image.asnumpy()
        else:
            images_trivial_augment_wide = np.append(images_trivial_augment_wide,
                                                    image.asnumpy(), axis=0)
    assert images_original.shape[0] == images_trivial_augment_wide.shape[0]
    if plot:
        visualize_list(images_original, images_trivial_augment_wide)

    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_trivial_augment_wide[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))

    # Trivial Augment Wide Images with SVHN num_magnitude_bins
    data_set3 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    trivial_augment_wide_op = TrivialAugmentWide(31, Inter.NEAREST, 20)
    transforms = [Decode(), Resize(size=[224, 224]), trivial_augment_wide_op]
    ds_trivial_augment_wide = data_set3.map(operations=transforms, input_columns="image")
    ds_trivial_augment_wide = ds_trivial_augment_wide.batch(512)
    for idx, (image, _) in enumerate(ds_trivial_augment_wide):
        if idx == 0:
            images_trivial_augment_wide = image.asnumpy()
        else:
            images_trivial_augment_wide = np.append(images_trivial_augment_wide,
                                                    image.asnumpy(), axis=0)
    assert images_original.shape[0] == images_trivial_augment_wide.shape[0]
    if plot:
        visualize_list(images_original, images_trivial_augment_wide)

    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_trivial_augment_wide[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))


def test_trivial_augment_wide_eager(plot=False):
    """
    Feature: TrivialAugmentWide
    Description: test TrivialAugmentWide eager
    Expectation: pass without error
    """
    image_file = "../data/dataset/testImageNetData/train/class1/1_1.jpg"
    img = np.fromfile(image_file, dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img = Decode()(img)
    img_trivial_augment_wided = TrivialAugmentWide(63)(img)

    if plot:
        visualize_image(img, img_trivial_augment_wided)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img_trivial_augment_wided),
                                                         img_trivial_augment_wided.shape))
    mse = diff_mse(img_trivial_augment_wided, img)
    logger.info("MSE= {}".format(str(mse)))


def test_trivial_augment_wide_invalid_input():
    """
    Feature: TrivialAugmentWide
    Description: test TrivialAugmentWide with invalid input
    Expectation: throw TypeError
    """
    try:
        image = np.random.randint(0, 256, (300, 300, 3)).astype(np.uint32)
        TrivialAugmentWide()(image)
    except RuntimeError as e:
        assert "TrivialAugmentWide: the data type of image tensor does not match the requirement of operator." in str(e)

    try:
        image = np.random.randint(0, 256, (300, 300, 1)).astype(np.uint8)
        TrivialAugmentWide()(image)
    except RuntimeError as e:
        assert "TrivialAugmentWide: the channel of image tensor does not match the requirement of operator" in str(e)

    try:
        image = np.random.randint(0, 256, (300, 300)).astype(np.uint8)
        TrivialAugmentWide()(image)
    except RuntimeError as e:
        assert "TrivialAugmentWide: the dimension of image tensor does not match the requirement of operator" in str(e)


def test_trivial_augment_wide_invalid_num_magnitude_bins():
    """
    Feature: TrivialAugmentWide
    Description: test TrivialAugmentWide with invalid num_magnitude_bins
    Expectation: throw TypeError
    """
    logger.info("test_trivial_augment_wide_invalid_num_magnitude_bins")
    data_dir = "../data/dataset/testImageNetData/train/"
    dataset = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    try:
        trivial_augment_wide_op = TrivialAugmentWide(num_magnitude_bins=-1)
        dataset.map(operations=trivial_augment_wide_op, input_columns=['image'])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input num_magnitude_bins is not within the required interval of [2, 16777216]." in str(e)


def test_trivial_augment_wide_invalid_interpolation():
    """
    Feature: TrivialAugmentWide
    Description: test TrivialAugmentWide with invalid interpolation
    Expectation: throw TypeError
    """
    logger.info("test_trivial_augment_wide_invalid_interpolation")
    data_dir = "../data/dataset/testImageNetData/train/"
    dataset = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    try:
        trivial_augment_wide_op = TrivialAugmentWide(interpolation="invalid")
        dataset.map(operations=trivial_augment_wide_op, input_columns=['image'])
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Argument interpolation with value invalid is not of type [<enum 'Inter'>]" in str(e)


def test_trivial_augment_wide_invalid_fill_value():
    """
    Feature: TrivialAugmentWide
    Description: test TrivialAugmentWide with invalid fill_value
    Expectation: throw TypeError or ValueError
    """
    logger.info("test_trivial_augment_wide_invalid_fill_value")
    data_dir = "../data/dataset/testImageNetData/train/"
    dataset = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    try:
        trivial_augment_wide_op = TrivialAugmentWide(fill_value=(10, 10))
        dataset.map(operations=trivial_augment_wide_op, input_columns=['image'])
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "fill_value should be a single integer or a 3-tuple." in str(e)
    try:
        trivial_augment_wide_op = TrivialAugmentWide(fill_value=300)
        dataset.map(operations=trivial_augment_wide_op, input_columns=['image'])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "is not within the required interval of [0, 255]." in str(e)


if __name__ == "__main__":
    test_trivial_augment_wide_pipeline(plot=True)
    test_trivial_augment_wide_eager(plot=True)
    test_trivial_augment_wide_invalid_input()
    test_trivial_augment_wide_invalid_num_magnitude_bins()
    test_trivial_augment_wide_invalid_interpolation()
    test_trivial_augment_wide_invalid_fill_value()
