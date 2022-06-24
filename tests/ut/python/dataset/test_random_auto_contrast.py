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
Testing RandomAutoContrast op in DE
"""
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import log as logger
from util import helper_random_op_pipeline, visualize_list, visualize_image, diff_mse

image_file = "../data/dataset/testImageNetData/train/class1/1_1.jpg"
data_dir = "../data/dataset/testImageNetData/train/"


def test_random_auto_contrast_pipeline(plot=False):
    """
    Feature: RandomAutoContrast op
    Description: Test RandomAutoContrast pipeline
    Expectation: Passes the test
    """
    logger.info("Test RandomAutoContrast pipeline")

    # Original Images
    images_original = helper_random_op_pipeline(data_dir)

    # Randomly Automatically Contrasted Images
    images_random_auto_contrast = helper_random_op_pipeline(
        data_dir, vision.RandomAutoContrast(0.6))

    if plot:
        visualize_list(images_original, images_random_auto_contrast)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_random_auto_contrast[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))


def test_random_auto_contrast_eager():
    """
    Feature: RandomAutoContrast op
    Description: Test RandomAutoContrast eager
    Expectation: Passes the test
    """
    img = np.fromfile(image_file, dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img = vision.Decode()(img)
    img_auto_contrast = vision.AutoContrast(1.0, None)(img)
    img_random_auto_contrast = vision.RandomAutoContrast(1.0, None, 1.0)(img)
    logger.info("Image.type: {}, Image.shape: {}".format(
        type(img_auto_contrast), img_random_auto_contrast.shape))

    assert img_auto_contrast.all() == img_random_auto_contrast.all()


def test_random_auto_contrast_comp(plot=False):
    """
    Feature: RandomAutoContrast op
    Description: Test RandomAutoContrast op compared with AutoContrast op
    Expectation: Resulting outputs from both operations are expected to be equal
    """
    random_auto_contrast_op = vision.RandomAutoContrast(prob=1.0)
    auto_contrast_op = vision.AutoContrast()

    dataset1 = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    for item in dataset1.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = item['image']
    dataset1.map(operations=random_auto_contrast_op, input_columns=['image'])
    dataset2 = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    dataset2.map(operations=auto_contrast_op, input_columns=['image'])
    for item1, item2 in zip(dataset1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            dataset2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_random_auto_contrast = item1['image']
        image_auto_contrast = item2['image']

    mse = diff_mse(image_auto_contrast, image_random_auto_contrast)
    assert mse == 0
    logger.info("mse: {}".format(mse))
    if plot:
        visualize_image(image, image_random_auto_contrast,
                        mse, image_auto_contrast)


def test_random_auto_contrast_invalid_prob():
    """
    Feature: RandomAutoContrast op
    Description: Test RandomAutoContrast with invalid prob parameter
    Expectation: Error is raised as expected
    """
    logger.info("test_random_auto_contrast_invalid_prob")
    dataset = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    try:
        random_auto_contrast_op = vision.RandomAutoContrast(prob=1.5)
        dataset = dataset.map(
            operations=random_auto_contrast_op, input_columns=['image'])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input prob is not within the required interval of [0.0, 1.0]." in str(
            e)


def test_random_auto_contrast_invalid_ignore():
    """
    Feature: RandomAutoContrast op
    Description: Test RandomAutoContrast with invalid ignore parameter
    Expectation: Error is raised as expected
    """
    logger.info("test_random_auto_contrast_invalid_ignore")
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data_set = data_set.map(operations=[vision.Decode(),
                                            vision.Resize((224, 224)),
                                            lambda img: np.array(img[:, :, 0])], input_columns=["image"])
        # invalid ignore
        data_set = data_set.map(operations=vision.RandomAutoContrast(
            ignore=255.5), input_columns="image")
    except TypeError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "Argument ignore with value 255.5 is not of type" in str(error)
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data_set = data_set.map(operations=[vision.Decode(), vision.Resize((224, 224)),
                                            lambda img: np.array(img[:, :, 0])], input_columns=["image"])
        # invalid ignore
        data_set = data_set.map(operations=vision.RandomAutoContrast(
            ignore=(10, 100)), input_columns="image")
    except TypeError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "Argument ignore with value (10,100) is not of type" in str(
            error)


def test_random_auto_contrast_invalid_cutoff():
    """
    Feature: RandomAutoContrast op
    Description: Test RandomAutoContrast with invalid cutoff parameter
    Expectation: Error is raised as expected
    """
    logger.info("test_random_auto_contrast_invalid_cutoff")
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data_set = data_set.map(operations=[vision.Decode(),
                                            vision.Resize((224, 224)),
                                            lambda img: np.array(img[:, :, 0])], input_columns=["image"])
        # invalid cutoff
        data_set = data_set.map(operations=vision.RandomAutoContrast(
            cutoff=-10.0), input_columns="image")
    except ValueError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "Input cutoff is not within the required interval of [0, 50)." in str(
            error)
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data_set = data_set.map(operations=[vision.Decode(),
                                            vision.Resize((224, 224)),
                                            lambda img: np.array(img[:, :, 0])], input_columns=["image"])
        # invalid cutoff
        data_set = data_set.map(operations=vision.RandomAutoContrast(
            cutoff=120.0), input_columns="image")
    except ValueError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "Input cutoff is not within the required interval of [0, 50)." in str(
            error)


def test_random_auto_contrast_one_channel():
    """
    Feature: RandomAutoContrast
    Description: Test with one channel images
    Expectation: Raise errors as expected
    """
    logger.info("test_random_auto_contrast_one_channel")

    c_op = vision.RandomAutoContrast()

    try:
        data_set = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data_set = data_set.map(operations=[vision.Decode(), vision.Resize((224, 224)),
                                            lambda img: np.array(img[:, :, 0])], input_columns=["image"])

        data_set = data_set.map(operations=c_op, input_columns="image")

    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "image shape is incorrect, expected num of channels is 3." in str(
            e)


def test_random_auto_contrast_four_dim():
    """
    Feature: RandomAutoContrast
    Description: Test with four dimension images
    Expectation: Raise errors as expected
    """
    logger.info("test_random_auto_contrast_four_dim")

    c_op = vision.RandomAutoContrast()

    try:
        data_set = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data_set = data_set.map(operations=[vision.Decode(), vision.Resize((224, 224)),
                                            lambda img: np.array(img[2, 200, 10, 32])], input_columns=["image"])

        data_set = data_set.map(operations=c_op, input_columns="image")

    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "image shape is not <H,W,C> or <H,W>" in str(e)


def test_random_auto_contrast_invalid_input():
    """
    Feature: RandomAutoContrast
    Description: Test with images in uint32 type
    Expectation: Raise errors as expected
    """
    logger.info("test_random_auto_contrast_invalid_input")

    c_op = vision.RandomAutoContrast()

    try:
        data_set = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data_set = data_set.map(operations=[vision.Decode(), vision.Resize((224, 224)),
                                            lambda img: np.array(img[2, 32, 3], dtype=uint32)], input_columns=["image"])
        data_set = data_set.map(operations=c_op, input_columns="image")

    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Cannot convert from OpenCV type, unknown CV type" in str(e)


if __name__ == "__main__":
    test_random_auto_contrast_pipeline(plot=True)
    test_random_auto_contrast_eager()
    test_random_auto_contrast_comp(plot=True)
    test_random_auto_contrast_invalid_prob()
    test_random_auto_contrast_invalid_ignore()
    test_random_auto_contrast_invalid_cutoff()
    test_random_auto_contrast_one_channel()
    test_random_auto_contrast_four_dim()
    test_random_auto_contrast_invalid_input()
