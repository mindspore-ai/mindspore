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
Testing RandomInvert in DE
"""
import numpy as np

import mindspore.dataset as ds
from mindspore.dataset.vision import Decode, Resize, RandomInvert, Invert
from mindspore import log as logger
from util import helper_random_op_pipeline, visualize_list, visualize_image, diff_mse

image_file = "../data/dataset/testImageNetData/train/class1/1_1.jpg"
data_dir = "../data/dataset/testImageNetData/train/"


def test_random_invert_pipeline(plot=False):
    """
    Feature: RandomInvert op
    Description: Test RandomInvert pipeline
    Expectation: Pipelines execute successfully
    """
    logger.info("Test RandomInvert pipeline")

    # Original Images
    images_original = helper_random_op_pipeline(data_dir)

    # Randomly Inverted Images
    images_random_invert = helper_random_op_pipeline(
        data_dir, RandomInvert(0.6))

    if plot:
        visualize_list(images_original, images_random_invert)

    num_samples = images_original.shape[0]
    mse = np.zeros(num_samples)
    for i in range(num_samples):
        mse[i] = diff_mse(images_random_invert[i], images_original[i])
    logger.info("MSE= {}".format(str(np.mean(mse))))


def test_random_invert_eager():
    """
    Feature: RandomInvert op
    Description: Test RandomInvert eager
    Expectation: The dataset is processed as expected
    """
    img = np.fromfile(image_file, dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img = Decode()(img)
    img_inverted = Invert()(img)
    img_random_inverted = RandomInvert(1.0)(img)
    logger.info("Image.type: {}, Image.shape: {}".format(
        type(img_random_inverted), img_random_inverted.shape))

    assert img_random_inverted.all() == img_inverted.all()


def test_random_invert_comp(plot=False):
    """
    Feature: RandomInvert op
    Description: Test RandomInvert op compared with Invert op
    Expectation: Resulting processed dataset is the same as expected
    """
    random_invert_op = RandomInvert(prob=1.0)
    invert_op = Invert()

    dataset1 = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    for item in dataset1.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = item['image']
    dataset1.map(operations=random_invert_op, input_columns=['image'])
    dataset2 = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    dataset2.map(operations=invert_op, input_columns=['image'])
    for item1, item2 in zip(dataset1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            dataset2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_random_inverted = item1['image']
        image_inverted = item2['image']

    mse = diff_mse(image_inverted, image_random_inverted)
    assert mse == 0
    logger.info("mse: {}".format(mse))
    if plot:
        visualize_image(image, image_random_inverted, mse, image_inverted)


def test_random_invert_invalid_prob():
    """
    Feature: RandomInvert op
    Description: Test invalid prob where prob is out of range
    Expectation: Error is raised as expected
    """
    logger.info("test_random_invert_invalid_prob")
    dataset = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    try:
        random_invert_op = RandomInvert(1.5)
        dataset = dataset.map(operations=random_invert_op,
                              input_columns=['image'])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input prob is not within the required interval of [0.0, 1.0]." in str(
            e)


def test_random_invert_one_channel():
    """
    Feature: RandomInvert
    Description: Test with one channel images
    Expectation: Raise errors as expected
    """
    logger.info("test_random_invert_one_channel")

    c_op = RandomInvert()

    try:
        data_set = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data_set = data_set.map(operations=[Decode(), Resize((224, 224)), lambda img: np.array(img[:, :, 0])],
                                input_columns=["image"])

        data_set = data_set.map(operations=c_op, input_columns="image")

    except RuntimeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "image shape is incorrect, expected num of channels is 3." in str(
            e)


def test_random_invert_four_dim():
    """
    Feature: RandomInvert
    Description: Test with four dimension images
    Expectation: Raise errors as expected
    """
    logger.info("test_random_invert_four_dim")

    c_op = RandomInvert()

    try:
        data_set = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data_set = data_set.map(operations=[Decode(), Resize((224, 224)), lambda img: np.array(img[2, 200, 10, 32])],
                                input_columns=["image"])

        data_set = data_set.map(operations=c_op, input_columns="image")

    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "image shape is not <H,W,C>" in str(e)


def test_random_invert_invalid_input():
    """
    Feature: RandomInvert
    Description: Test with images in uint32 type
    Expectation: Raise errors as expected
    """
    logger.info("test_random_invert_invalid_input")

    c_op = RandomInvert()

    try:
        data_set = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
        data_set = data_set.map(operations=[Decode(), Resize((224, 224)),
                                            lambda img: np.array(img[2, 32, 3], dtype=uint32)], input_columns=["image"])
        data_set = data_set.map(operations=c_op, input_columns="image")

    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Cannot convert from OpenCV type, unknown CV type" in str(e)


if __name__ == "__main__":
    test_random_invert_pipeline(plot=True)
    test_random_invert_eager()
    test_random_invert_comp(plot=True)
    test_random_invert_invalid_prob()
    test_random_invert_one_channel()
    test_random_invert_four_dim()
    test_random_invert_invalid_input()
