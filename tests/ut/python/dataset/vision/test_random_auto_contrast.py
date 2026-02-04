# Copyright 2025 Huawei Technologies Co., Ltd
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
import os
import pytest

import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger
from util import helper_random_op_pipeline, visualize_list, visualize_image, diff_mse

image_file = "../data/dataset/testImageNetData/train/class1/1_1.jpg"
data_dir = "../data/dataset/testImageNetData/train/"
TEST_DATA_DATASET_FUNC ="../data/dataset/"

CONTRAST_PERF_FILE = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "pen.jpg")


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


def test_random_auto_contrast_operation_01():
    """
    Feature: RandomAutoContrast operation
    Description: Testing the normal functionality of the RandomAutoContrast operator
    Expectation: The Output is equal to the expected output
    """
    # Eager mode, no parameters set, RandomAutoContrast interface invocation successful
    image = np.fromfile(CONTRAST_PERF_FILE, dtype=np.int32)
    op1 = vision.Decode()
    op2 = vision.RandomAutoContrast()
    image_origin = op1(image)
    _ = op2(image_origin)

    # When the parameter ignore is set to [1], the RandomAutoContrast interface call succeeds.
    image = np.fromfile(CONTRAST_PERF_FILE, dtype=np.int32)
    op1 = vision.Decode()
    op2 = vision.RandomAutoContrast(ignore=[1])
    image_origin = op1(image)
    _ = op2(image_origin)

    # When the parameter prob is 0, the RandomAutoContrast interface call succeeds.
    image = np.fromfile(CONTRAST_PERF_FILE, dtype=np.int32)
    op1 = vision.Decode()
    op2 = vision.RandomAutoContrast(prob=0)
    image_origin = op1(image)
    _ = op2(image_origin)

    # When the parameter prob is 1, the RandomAutoContrast interface call succeeds.
    image = np.fromfile(CONTRAST_PERF_FILE, dtype=np.int32)
    op1 = vision.Decode()
    op2 = vision.RandomAutoContrast(prob=1, ignore=[0, 0])
    image_origin = op1(image)
    _ = op2(image_origin)

    # When the parameter ignore is set to 0, the RandomAutoContrast interface call succeeds.
    image = np.fromfile(CONTRAST_PERF_FILE, dtype=np.int32)
    op1 = vision.Decode()
    op2 = vision.RandomAutoContrast(ignore=0)
    image_origin = op1(image)
    _ = op2(image_origin)

    # When the parameter ignore is [1, 234], the RandomAutoContrast interface call succeeds.
    image = np.fromfile(CONTRAST_PERF_FILE, dtype=np.int32)
    op1 = vision.Decode()
    op2 = vision.RandomAutoContrast(ignore=[1, 234])
    image_origin = op1(image)
    _ = op2(image_origin)

    # When setting default parameter values, the RandomAutoContrast interface call succeeds.
    image = np.fromfile(CONTRAST_PERF_FILE, dtype=np.int32)
    op1 = vision.Decode()
    op2 = vision.RandomAutoContrast(cutoff=0.0, ignore=None, prob=0.5)
    image_origin = op1(image)
    _ = op2(image_origin)

    # When the parameter cutoff is obtained via randint, the RandomAutoContrast interface call succeeds.
    image = np.fromfile(CONTRAST_PERF_FILE, dtype=np.int32)
    op1 = vision.Decode()
    op2 = vision.RandomAutoContrast(cutoff=np.random.randint(2))
    image_origin = op1(image)
    _ = op2(image_origin)

    # The input image shape is < H, W >. The RandomAutoContrast interface call succeeded.
    image = np.random.randint(0, 255, (658, 714)).astype(np.uint8)
    op = vision.RandomAutoContrast(prob=0.2)
    _ = op(image)


def test_random_auto_contrast_exception_01():
    """
    Feature: RandomAutoContrast operation
    Description: Testing the RandomAutoContrast Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When the parameter prob is greater than 1, the RandomAutoContrast interface call fails.
    image = np.fromfile(CONTRAST_PERF_FILE, dtype=np.int32)
    op1 = vision.Decode()
    rgb_data = op1(image)
    with pytest.raises(ValueError, match='Input prob is not within the required interval'):
        op2 = vision.RandomAutoContrast(prob=2)
        op2(rgb_data)

    # When the parameter prob is a list, the RandomAutoContrast interface call fails.
    image = np.fromfile(CONTRAST_PERF_FILE, dtype=np.int32)
    op1 = vision.Decode()
    rgb_data = op1(image)
    with pytest.raises(TypeError, match="is not of type"):
        op2 = vision.RandomAutoContrast(prob=[2])
        op2(rgb_data)

    # When the parameter prob is a tuple, the RandomAutoContrast interface call fails.
    image = np.fromfile(CONTRAST_PERF_FILE, dtype=np.int32)
    op1 = vision.Decode()
    rgb_data = op1(image)
    with pytest.raises(TypeError, match="is not of type"):
        op2 = vision.RandomAutoContrast(prob=(2,))
        op2(rgb_data)

    # When the parameter prob is a string, the RandomAutoContrast interface call fails.
    image = np.fromfile(CONTRAST_PERF_FILE, dtype=np.int32)
    op1 = vision.Decode()
    rgb_data = op1(image)
    with pytest.raises(TypeError, match="is not of type"):
        op2 = vision.RandomAutoContrast(prob='1')
        op2(rgb_data)

    # When the parameter prob is a bool, the RandomAutoContrast interface call fails.
    image = np.fromfile(CONTRAST_PERF_FILE, dtype=np.int32)
    op1 = vision.Decode()
    rgb_data = op1(image)
    with pytest.raises(TypeError, match="is not of type"):
        op2 = vision.RandomAutoContrast(prob=False)
        op2(rgb_data)

    # When the cutoff parameter is negative, the RandomAutoContrast interface call fails.
    image = np.fromfile(CONTRAST_PERF_FILE, dtype=np.int32)
    op1 = vision.Decode()
    rgb_data = op1(image)
    with pytest.raises(ValueError, match="Input cutoff is not within the required interval"):
        op2 = vision.RandomAutoContrast(cutoff=-10, prob=1.0)
        op2(rgb_data)

    # When the parameter ignore exceeds 255, the RandomAutoContrast interface call fails.
    image = np.fromfile(CONTRAST_PERF_FILE, dtype=np.int32)
    op1 = vision.Decode()
    rgb_data = op1(image)
    with pytest.raises(ValueError, match="Input ignore is not within the required interval"):
        op2 = vision.RandomAutoContrast(cutoff=10, ignore=[1, 300], prob=1.0)
        op2(rgb_data)

    # When the parameter prob is 1.1, the RandomAutoContrast interface call fails.
    with pytest.raises(ValueError) as error_info:
        err_msg = "Input prob is not within the required interval of [0.0, 1.0]."
        vision.RandomAutoContrast(prob=1.1)
    assert err_msg in str(error_info.value)

    # When the parameter prob is negative, the RandomAutoContrast interface call fails.
    with pytest.raises(ValueError) as error_info:
        err_msg = "Input prob is not within the required interval of [0.0, 1.0]."
        vision.RandomAutoContrast(prob=-0.1)
    assert err_msg in str(error_info.value)

    # When the cutoff parameter is negative, the RandomAutoContrast interface call fails.
    with pytest.raises(ValueError) as error_info:
        err_msg = "Input cutoff is not within the required interval of [0, 50)."
        vision.RandomAutoContrast(cutoff=-0.1)
    assert err_msg in str(error_info.value)

    # When the cutoff parameter is set to 50, the RandomAutoContrast interface call fails.
    with pytest.raises(ValueError) as error_info:
        err_msg = "Input cutoff is not within the required interval of [0, 50)."
        vision.RandomAutoContrast(cutoff=50)
    assert err_msg in str(error_info.value)

    # When the cutoff parameter is a string, the RandomAutoContrast interface call fails.
    with pytest.raises(TypeError) as error_info:
        err_msg = "Argument cutoff with value 0 is not of type [<class 'int'>, <class 'float'>], but got <class 'str'>."
        vision.RandomAutoContrast(cutoff="0")
    assert err_msg in str(error_info.value)

    # When the cutoff parameter is a list, the RandomAutoContrast interface call fails.
    with pytest.raises(TypeError) as error_info:
        err_msg = "Argument cutoff with value [0] is not of type [<class 'int'>, <class 'float'>], but got <class 'l" \
                  "ist'>."
        vision.RandomAutoContrast(cutoff=[0])
    assert err_msg in str(error_info.value)

    # When the parameter "ignore" is set to "float", the "RandomAutoContrast" interface call fails.
    with pytest.raises(TypeError) as error_info:
        err_msg = "Argument ignore with value 0.1 is not of type [<class 'list'>, <class 'tuple'>, <class 'int'>], " \
                  "but got <class 'float'>."
        vision.RandomAutoContrast(ignore=0.1)
    assert err_msg in str(error_info.value)


def test_random_auto_contrast_exception_02():
    """
    Feature: RandomAutoContrast operation
    Description: Testing the RandomAutoContrast Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When the ignore parameter is negative, the RandomAutoContrast interface call fails.
    with pytest.raises(ValueError) as error_info:
        err_msg = "Input ignore is not within the required interval of [0, 255]."
        vision.RandomAutoContrast(ignore=-1)
    assert err_msg in str(error_info.value)

    # When the parameter ignore exceeds 255, the RandomAutoContrast interface call fails.
    with pytest.raises(ValueError) as error_info:
        err_msg = "Input ignore is not within the required interval of [0, 255]."
        vision.RandomAutoContrast(ignore=256)
    assert err_msg in str(error_info.value)

    # When the parameter `ignore` is a string, the RandomAutoContrast interface call fails.
    with pytest.raises(TypeError, match="Argument ignore with value 1 is not of type \\[<class 'list'>,"
                                        " <class 'tuple'>, <class 'int'>\\], but got <class 'str'>."):
        vision.RandomAutoContrast(ignore="1")

    # When input data is a list, the RandomAutoContrast interface call fails.
    image = np.fromfile(CONTRAST_PERF_FILE, dtype=np.int32).tolist()
    op1 = vision.Decode()
    with pytest.raises(TypeError, match="The type of the encoded image should be <class 'numpy.ndarray'>,"
                                        " but got <class 'list'>."):
        rgb_data = op1(image)
        op2 = vision.RandomAutoContrast(prob=2)
        op2(rgb_data)

    # Input image shape is not < H, W > or < H, W, C >; RandomAutoContrast interface call failed.
    op = vision.RandomAutoContrast(prob=0.8)
    with pytest.raises(RuntimeError, match="RandomAutoContrast: image shape is not <H,W,C> or <H,W>,"
                                           " got rank: 0"):
        op(np.array(10))

    # When the input data channel is not 3, the RandomAutoContrast interface call fails.
    image = np.random.randint(0, 255, (658, 714, 4)).astype(np.uint8)
    op = vision.RandomAutoContrast(prob=0.3)
    with pytest.raises(RuntimeError, match="RandomAutoContrast: image shape is incorrect, "
                                           "expected num of channels is 3, but got: 4"):
        op(image)

    # The input image's shape is not <H, W> or <H, W, C>. The RandomAutoContrast interface call failed.
    image = np.random.randint(0, 255, (658, 714, 10, 3)).astype(np.uint8)
    op = vision.RandomAutoContrast(prob=0.3)
    with pytest.raises(RuntimeError, match="RandomAutoContrast: image shape is not <H,W,C> or <H,W>,"
                                           " got rank: 4"):
        op(image)


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
    test_random_auto_contrast_operation_01()
    test_random_auto_contrast_exception_01()
    test_random_auto_contrast_exception_02()
