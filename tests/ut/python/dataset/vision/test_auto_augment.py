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
Testing AutoAugment in DE
"""
import numpy as np
import os
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as v_trans
from mindspore.dataset.vision import Decode, AutoAugment
from mindspore.dataset.vision.utils import AutoAugmentPolicy, Inter
from mindspore import log as logger
from util import helper_random_op_pipeline, visualize_image, visualize_list, diff_mse


image_file = "../data/dataset/testImageNetData/train/class1/1_1.jpg"
data_dir = "../data/dataset/testImageNetData/train/"
TEST_DATA_DATASET_FUNC ="../data/dataset/"


def dir_data():
    """Obtain the dataset"""
    data_list = []
    data_dir1 = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train/")
    data_dir2 = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train", "class1", "1_1.jpg")
    data_list.append(data_dir1)
    data_list.append(data_dir2)
    return data_list


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


def test_auto_augment_operation_01():
    """
    Feature: AutoAugment operation
    Description: Testing the normal functionality of the AutoAugment operator
    Expectation: The Output is equal to the expected output
    """
    # AutoAugment Operator Normal Functionality: Testing Pipeline Mode
    dataset1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    policy = AutoAugmentPolicy.IMAGENET
    interpolation = Inter.NEAREST
    fill_value = (10, 100, 150)
    auto_augment_op = [v_trans.Decode(to_pil=False),
                       v_trans.AutoAugment(policy=policy, interpolation=interpolation, fill_value=fill_value)]
    dataset2 = dataset2.map(input_columns=["image"], operations=auto_augment_op)
    for _ in zip(dataset1.create_dict_iterator(output_numpy=True), dataset2.create_dict_iterator(output_numpy=True)):
        pass

    # AutoAugment Operator Normal Functionality: Testing Pipeline Mode
    dataset1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    policy = AutoAugmentPolicy.CIFAR10
    interpolation = Inter.BILINEAR
    fill_value = 0
    auto_augment_op = [v_trans.Decode(to_pil=False),
                       v_trans.AutoAugment(policy=policy, interpolation=interpolation, fill_value=fill_value)]
    dataset2 = dataset2.map(input_columns=["image"], operations=auto_augment_op)
    for _ in zip(dataset1.create_dict_iterator(output_numpy=True), dataset2.create_dict_iterator(output_numpy=True)):
        pass

    # AutoAugment Operator Normal Functionality: Interpolation parameter set to Inter.BICUBIC
    with Image.open(dir_data()[1]) as image:
        policy = AutoAugmentPolicy.SVHN
        interpolation = Inter.BICUBIC
        fill_value = 255
        auto_augment_op = v_trans.AutoAugment(policy=policy, interpolation=interpolation, fill_value=fill_value)
        _ = auto_augment_op(image)

    # AutoAugment Operator Normal Functionality: Interpolation parameter is Inter.AREA
    with Image.open(dir_data()[1]) as image:
        policy = AutoAugmentPolicy.IMAGENET
        interpolation = Inter.AREA
        fill_value = (0, 100, 255)
        auto_augment_op = v_trans.AutoAugment(policy=policy, interpolation=interpolation, fill_value=fill_value)
        _ = auto_augment_op(image)

    # AutoAugment Operator Normal Functionality: Interpolation parameter is Inter.NEAREST
    image = np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8)
    policy = AutoAugmentPolicy.CIFAR10
    interpolation = Inter.NEAREST
    fill_value = (128, 0, 250)
    auto_augment_op = v_trans.AutoAugment(policy=policy, interpolation=interpolation, fill_value=fill_value)
    _ = auto_augment_op(image)


def test_auto_augment_exception_01():
    """
    Feature: AutoAugment operation
    Description: Testing the AutoAugment Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # AutoAugment Exception Scenario: Input is a numpy array with channel depth of 4
    image = np.random.randint(0, 255, (100, 100, 4)).astype(np.uint8)
    auto_augment_op = v_trans.AutoAugment()
    with pytest.raises(RuntimeError, match="AutoAugment: channel of input image should be 3, but got: 4"):
        auto_augment_op(image)

    # AutoAugment Exception Scenario: Inputting 4-Dimensional NumPy Data
    image = np.random.randint(0, 255, (658, 714, 3, 3)).astype(np.uint8)
    auto_augment_op = v_trans.AutoAugment()
    with pytest.raises(RuntimeError, match="AutoAugment: input tensor is not in shape of <H,W,C>, but got rank: 4"):
        auto_augment_op(image)

    # AutoAugment Exception Scenario: Input shape is 0
    image = np.array(10)
    auto_augment_op = v_trans.AutoAugment()
    with pytest.raises(RuntimeError, match="AutoAugment: input tensor is not in shape of <H,W,C>, but got rank: 0"):
        auto_augment_op(image)

    # AutoAugment Exception Scenario: Input is a list [numpy, numpy, ...]
    image = list(np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8))
    auto_augment_op = v_trans.AutoAugment()
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        auto_augment_op(image)

    # AutoAugment Anomaly Scenario: No Input Data
    auto_augment_op = v_trans.AutoAugment()
    with pytest.raises(RuntimeError, match="Input Tensor is not valid."):
        auto_augment_op()

    # AutoAugment Anomaly Scenario: Two Inputs
    image = np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8)
    auto_augment_op = v_trans.AutoAugment()
    with pytest.raises(RuntimeError, match="The op is OneToOne, can only accept one tensor as input."):
        auto_augment_op(image, image)

    # AutoAugment Exception Scenario: policy parameter is a string
    with pytest.raises(TypeError, match="Argument policy with value IMAGENET is not of type \\[<enum"
                                        " 'AutoAugmentPolicy'>\\], but got <class 'str'>."):
        v_trans.AutoAugment("IMAGENET")

    # AutoAugment Exception Scenario: policy parameter is int
    with pytest.raises(TypeError, match="Argument policy with value 1 is not of type \\[<enum"
                                        " 'AutoAugmentPolicy'>\\], but got <class 'int'>."):
        v_trans.AutoAugment(1)

    # AutoAugment Exception Scenario: policy parameter is None
    with pytest.raises(TypeError, match="Argument policy with value None is not of type \\[<enum"
                                        " 'AutoAugmentPolicy'>\\], but got <class 'NoneType'>."):
        v_trans.AutoAugment(None)

    # AutoAugment Exception Scenario: interpolation parameter is a string
    with pytest.raises(TypeError, match="Argument interpolation with value AREA is not of type \\[<enum"
                                        " 'Inter'>\\], but got <class 'str'>."):
        v_trans.AutoAugment(AutoAugmentPolicy.IMAGENET, "AREA")

    # AutoAugment Exception Scenario: interpolation parameter is list
    with pytest.raises(TypeError, match="Argument interpolation with value \\[<Inter.AREA: 4>\\] is not"
                                        " of type \\[<enum 'Inter'>\\], but got <class 'list'>."):
        v_trans.AutoAugment(AutoAugmentPolicy.IMAGENET, [Inter.AREA])

    # AutoAugment Exception Scenario: interpolation parameter is int
    with pytest.raises(TypeError, match="Argument interpolation with value 0 is not of "
                                        "type \\[<enum 'Inter'>\\], but got <class 'int'>."):
        v_trans.AutoAugment(AutoAugmentPolicy.IMAGENET, 0)

    # AutoAugment Exception Scenario: fill_value parameter is less than 0
    with pytest.raises(ValueError, match="Input fill_value is not within the required interval of \\[0, 255\\]."):
        v_trans.AutoAugment(AutoAugmentPolicy.IMAGENET, Inter.AREA, -1)

    # AutoAugment Anomaly Scenario: fill_value exceeds 255
    with pytest.raises(ValueError,
                       match="Input fill_value\\[2\\] is not within the required interval of \\[0, 255\\]."):
        v_trans.AutoAugment(AutoAugmentPolicy.IMAGENET, Inter.AREA, (0, 255, 256))

    # AutoAugment Exception Scenario: fill_value parameter is of type float
    with pytest.raises(TypeError,
                       match="Argument fill_value\\[2\\] with value 25.0 is not of "
                             "type \\[<class 'int'>\\], but got <class 'float'>."):
        v_trans.AutoAugment(AutoAugmentPolicy.IMAGENET, Inter.AREA, (0, 255, 25.0))

    # AutoAugment Exception Scenario: fill_value parameter is of type str
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        v_trans.AutoAugment(AutoAugmentPolicy.IMAGENET, Inter.AREA, "1")

    # AutoAugment Exception Scenario: fill_value parameter is of type list
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        v_trans.AutoAugment(AutoAugmentPolicy.IMAGENET, Inter.AREA, [1, 2, 3])


if __name__ == "__main__":
    test_auto_augment_pipeline(plot=True)
    test_auto_augment_eager(plot=True)
    test_auto_augment_invalid_policy()
    test_auto_augment_invalid_interpolation()
    test_auto_augment_invalid_fill_value()
    test_auto_augment_operation_01()
    test_auto_augment_exception_01()
