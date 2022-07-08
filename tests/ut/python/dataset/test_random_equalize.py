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
Testing RandomEqualize op in DE
"""
import numpy as np

import mindspore.dataset as ds
from mindspore.dataset.vision import Decode, Resize, RandomEqualize, Equalize
from mindspore import log as logger
from util import helper_random_op_pipeline, visualize_list, visualize_image, diff_mse

image_file = "../data/dataset/testImageNetData/train/class1/1_1.jpg"
data_dir = "../data/dataset/testImageNetData/train/"


def test_random_equalize_pipeline(plot=False):
    """
    Feature: RandomEqualize op
    Description: Test RandomEqualize pipeline
    Expectation: Passes the test
    """
    logger.info("Test RandomEqualize pipeline")

    # Original Images
    images_original = helper_random_op_pipeline(data_dir)

    # Randomly Equalized Images
    images_random_equalize = helper_random_op_pipeline(
        data_dir, RandomEqualize(0.6))

    if plot:
        visualize_list(images_original, images_random_equalize)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_random_equalize[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))


def test_random_equalize_eager():
    """
    Feature: RandomEqualize op
    Description: Test RandomEqualize eager
    Expectation: Passes the test
    """
    img = np.fromfile(image_file, dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img = Decode()(img)
    img_equalized = Equalize()(img)
    img_random_equalized = RandomEqualize(1.0)(img)
    logger.info("Image.type: {}, Image.shape: {}".format(
        type(img_random_equalized), img_random_equalized.shape))

    assert img_random_equalized.all() == img_equalized.all()


def test_random_equalize_comp(plot=False):
    """
    Feature: RandomEqualize op
    Description: Test RandomEqualize op with Equalize op
    Expectation: Resulting outputs from both ops are the same as expected
    """
    random_equalize_op = RandomEqualize(prob=1.0)
    equalize_op = Equalize()

    dataset1 = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    for item in dataset1.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = item['image']
    dataset1.map(operations=random_equalize_op, input_columns=['image'])
    dataset2 = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    dataset2.map(operations=equalize_op, input_columns=['image'])
    for item1, item2 in zip(dataset1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            dataset2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_random_equalized = item1['image']
        image_equalized = item2['image']

    mse = diff_mse(image_equalized, image_random_equalized)
    assert mse == 0
    logger.info("mse: {}".format(mse))
    if plot:
        visualize_image(image, image_random_equalized, mse, image_equalized)


def test_random_equalize_invalid_prob():
    """
    Feature: RandomEqualize op
    Description: Test RandomEqualize eager with prob out of range
    Expectation: Error is raised as expected
    """
    logger.info("test_random_equalize_invalid_prob")
    dataset = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    try:
        random_equalize_op = RandomEqualize(1.5)
        dataset = dataset.map(
            operations=random_equalize_op, input_columns=['image'])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input prob is not within the required interval of [0.0, 1.0]." in str(e)


def test_random_equalize_four_channel():
    """
    Feature: RandomEqualize
    Description: test with four channel images
    Expectation: raise errors as expected
    """
    logger.info("test_random_equalize_four_channel")

    c_op = RandomEqualize()

    try:
        data_set = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data_set = data_set.map(operations=[Decode(), Resize((224, 224)),
                                            lambda img: np.array(img[128, 98, 4])], input_columns=["image"])

        data_set = data_set.map(operations=c_op, input_columns="image")

    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "image shape is incorrect, expected num of channels is 1 or 3." in str(e)


def test_random_equalize_four_dim():
    """
    Feature: RandomEqualize
    Description: test with four dimension images
    Expectation: raise errors as expected
    """
    logger.info("test_random_equalize_four_dim")

    c_op = RandomEqualize()

    try:
        data_set = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data_set = data_set.map(operations=[Decode(), Resize((224, 224)),
                                            lambda img: np.array(img[2, 200, 10, 32])], input_columns=["image"])

        data_set = data_set.map(operations=c_op, input_columns="image")

    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "image shape is not <H,W,C> or <H,W> " in str(e)


def test_random_equalize_invalid_input():
    """
    Feature: RandomEqualize
    Description: test with images in uint32 type
    Expectation: raise errors as expected
    """
    logger.info("test_random_equalize_invalid_input")

    c_op = RandomEqualize()

    try:
        data_set = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data_set = data_set.map(
            operations=[Decode(), Resize((224, 224)),
                        lambda img: np.array(img[2, 32, 3], dtype=float32)],
            input_columns=["image"])
        data_set = data_set.map(operations=c_op, input_columns="image")

    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input Tensor type should be uint8, got type is non-uint8." in str(e)


if __name__ == "__main__":
    test_random_equalize_pipeline(plot=True)
    test_random_equalize_eager()
    test_random_equalize_comp(plot=True)
    test_random_equalize_invalid_prob()
    test_random_equalize_four_channel()
    test_random_equalize_four_dim()
    test_random_equalize_invalid_input()
