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
Testing RandomAdjustSharpness in DE
"""
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import log as logger
from util import helper_random_op_pipeline, visualize_list, visualize_image, diff_mse

image_file = "../data/dataset/testImageNetData/train/class1/1_1.jpg"
data_dir = "../data/dataset/testImageNetData/train/"


def test_random_adjust_sharpness_pipeline(plot=False):
    """
    Feature: RandomAdjustSharpness op
    Description: Test RandomAdjustSharpness pipeline
    Expectation: Passes the test
    """
    logger.info("Test RandomAdjustSharpness pipeline")

    # Original Images
    images_original = helper_random_op_pipeline(data_dir)

    # Randomly Adjust Sharpness Images
    images_random_adjust_sharpness = helper_random_op_pipeline(
        data_dir, vision.RandomAdjustSharpness(2.0, 0.6))

    if plot:
        visualize_list(images_original, images_random_adjust_sharpness)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_random_adjust_sharpness[i],
                          images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))


def test_random_adjust_sharpness_eager():
    """
    Feature: RandomAdjustSharpness op
    Description: Test RandomAdjustSharpness eager
    Expectation: Passes the equality test
    """
    img = np.fromfile(image_file, dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img = vision.Decode()(img)
    img_sharped = vision.RandomSharpness((2.0, 2.0))(img)
    img_random_sharped = vision.RandomAdjustSharpness(2.0, 1.0)(img)
    logger.info("Image.type: {}, Image.shape: {}".format(
        type(img_random_sharped), img_random_sharped.shape))

    assert img_random_sharped.all() == img_sharped.all()


def test_random_adjust_sharpness_comp(plot=False):
    """
    Feature: RandomAdjustSharpness op
    Description: Test RandomAdjustSharpness op compared with Sharpness op
    Expectation: Resulting outputs from both operations are expected to be equal
    """
    random_adjust_sharpness_op = vision.RandomAdjustSharpness(
        degree=2.0, prob=1.0)
    sharpness_op = vision.RandomSharpness((2.0, 2.0))

    dataset1 = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    for item in dataset1.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = item['image']
    dataset1.map(operations=random_adjust_sharpness_op,
                 input_columns=['image'])
    dataset2 = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    dataset2.map(operations=sharpness_op, input_columns=['image'])

    for item1, item2 in zip(dataset1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            dataset2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_random_sharpness = item1['image']
        image_sharpness = item2['image']

    mse = diff_mse(image_sharpness, image_random_sharpness)
    assert mse == 0
    logger.info("mse: {}".format(mse))
    if plot:
        visualize_image(image, image_random_sharpness, mse, image_sharpness)


def test_random_adjust_sharpness_invalid_prob():
    """
    Feature: RandomAdjustSharpness op
    Description: Test invalid prob where prob is out of range
    Expectation: Error is raised as expected
    """
    logger.info("test_random_adjust_sharpness_invalid_prob")
    dataset = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    try:
        random_adjust_sharpness_op = vision.RandomAdjustSharpness(2.0, 1.5)
        dataset = dataset.map(
            operations=random_adjust_sharpness_op, input_columns=['image'])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input prob is not within the required interval of [0.0, 1.0]." in str(
            e)


def test_random_adjust_sharpness_invalid_degree():
    """
    Feature: RandomAdjustSharpness op
    Description: Test invalid prob where prob is out of range
    Expectation: Error is raised as expected
    """
    logger.info("test_random_adjust_sharpness_invalid_prob")
    dataset = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    try:
        random_adjust_sharpness_op = vision.RandomAdjustSharpness(-1.0, 1.5)
        dataset = dataset.map(
            operations=random_adjust_sharpness_op, input_columns=['image'])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "interval" in str(e)


def test_random_adjust_sharpness_four_dim():
    """
    Feature: RandomAdjustSharpness
    Description: test with four dimension images
    Expectation: raise errors as expected
    """
    logger.info("test_random_adjust_sharpness_four_dim")

    c_op = vision.RandomAdjustSharpness(2.0, 0.5)

    try:
        data_set = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data_set = data_set.map(operations=[vision.Decode(), vision.Resize((224, 224)),
                                            lambda img: np.array(img[2, 200, 10, 32])], input_columns=["image"])

        data_set = data_set.map(operations=c_op, input_columns="image")

    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "image shape is not <H,W,C>" in str(e)


def test_random_adjust_sharpness_invalid_input():
    """
    Feature: RandomAdjustSharpness
    Description: test with images in uint32 type
    Expectation: raise errors as expected
    """
    logger.info("test_random_adjust_sharpness_invalid_input")

    c_op = vision.RandomAdjustSharpness(2.0, 0.5)

    try:
        data_set = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data_set = data_set.map(operations=[vision.Decode(), vision.Resize((224, 224)),
                                            lambda img: np.array(img[2, 32, 3], dtype=uint32)], input_columns=["image"])
        data_set = data_set.map(operations=c_op, input_columns="image")

    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Cannot convert from OpenCV type, unknown CV type" in str(e)


if __name__ == "__main__":
    test_random_adjust_sharpness_pipeline(plot=True)
    test_random_adjust_sharpness_eager()
    test_random_adjust_sharpness_comp(plot=True)
    test_random_adjust_sharpness_invalid_prob()
    test_random_adjust_sharpness_invalid_degree()
    test_random_adjust_sharpness_four_dim()
    test_random_adjust_sharpness_invalid_input()
