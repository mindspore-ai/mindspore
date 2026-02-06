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
Testing RandAugment in DE
"""
import numpy as np
import os
import pytest
from PIL import Image

import mindspore.dataset as ds
from mindspore.dataset.vision import Decode, RandAugment, Resize, Inter
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger
from util import visualize_image, visualize_list, diff_mse

IMAGE_FILE = "../data/dataset/testImageNetData/train/class1/1_1.jpg"
DATA_DIR = "../data/dataset/testImageNetData/train/"
TEST_DATA_DATASET_FUNC ="../data/dataset/"


data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
image_file = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train", "class1", "1_1.jpg")


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


def test_rand_augment_operation_01():
    """
    Feature: RandAugment operation
    Description: Testing the normal functionality of the RandAugment operator
    Expectation: The Output is equal to the expected output
    """
    # test default value
    dataset1 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    randaugment_op = vision.RandAugment()
    dataset2 = dataset2.map(input_columns=["image"], operations=randaugment_op)
    num_iter = 0
    for _ in zip(dataset1.create_dict_iterator(output_numpy=True),
                 dataset2.create_dict_iterator(output_numpy=True)):
        num_iter += 1
    assert num_iter == 2

    # test fill_value 255
    dataset1 = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(data_dir, 1, shuffle=False, decode=True)
    num_ops = 3
    magnitude = 13
    num_magnitude_bins = 50
    interpolation = Inter.AREA
    fill_value = (255, 255, 255)
    randaugment_op = vision.RandAugment(num_ops=num_ops, magnitude=magnitude, num_magnitude_bins=num_magnitude_bins,
                                        interpolation=interpolation, fill_value=fill_value)
    dataset2 = dataset2.map(input_columns=["image"], operations=randaugment_op)
    num_iter = 0
    for _ in zip(dataset1.create_dict_iterator(output_numpy=True),
                 dataset2.create_dict_iterator(output_numpy=True)):
        num_iter += 1
    assert num_iter == 1

    # Test eager. Normal test.magnitude less than num_magnitude_bins
    image = np.fromfile(image_file, dtype=np.int32)
    op1 = vision.Decode()
    num_ops = 13
    magnitude = 2
    num_magnitude_bins = 3
    interpolation = Inter.BILINEAR
    fill_value = 1
    op2 = vision.RandAugment(num_ops=num_ops, magnitude=magnitude, num_magnitude_bins=num_magnitude_bins,
                             interpolation=interpolation, fill_value=fill_value)
    image_origin = op1(image)
    _ = op2(image_origin)

    # Test eager. default param test.
    image = np.fromfile(image_file, dtype=np.int32)
    op1 = vision.Decode()
    op2 = vision.RandAugment()
    image_origin = op1(image)
    _ = op2(image_origin)

    # input is pillow image
    image = Image.open(image_file)
    interpolation = Inter.BILINEAR
    fill_value = 255
    randaugment_op = vision.RandAugment(interpolation=interpolation, fill_value=fill_value)
    _ = randaugment_op(image)
    image.close()

    # Test eager. fill_value param test.
    image = Image.open(image_file)
    num_ops = 300
    magnitude = 1
    num_magnitude_bins = 15
    interpolation = Inter.AREA
    fill_value = (0, 100, 255)
    randaugment_op = vision.RandAugment(num_ops=num_ops, magnitude=magnitude, num_magnitude_bins=num_magnitude_bins,
                                        interpolation=interpolation, fill_value=fill_value)
    _ = randaugment_op(image)
    image.close()

    # input is numpy
    image = np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8)
    interpolation = Inter.BICUBIC
    fill_value = (128, 0, 250)
    randaugment_op = vision.RandAugment(interpolation=interpolation, fill_value=fill_value)
    _ = randaugment_op(image)


def test_rand_augment_exception_01():
    """
    Feature: RandAugment operation
    Description: Testing the RandAugment Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # test input is list, do not supported
    image = list(np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8))
    interpolation = Inter.BICUBIC
    fill_value = (128, 0, 250)
    randaugment_op = vision.RandAugment(interpolation=interpolation, fill_value=fill_value)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        randaugment_op(image)

    # test channels is not 3
    image = np.random.randint(0, 255, (100, 100, 4)).astype(np.uint8)
    randaugment_op = vision.RandAugment()
    with pytest.raises(RuntimeError,
                       match=".*RandAugment: the channel of image tensor does not match the requirement of operator."
                             " Expecting tensor in channel of \\(3\\). But got channel 4..*"):
        randaugment_op(image)

    # test image shape is [658, 714, 3, 3]
    image = np.random.randint(0, 255, (658, 714, 3, 3)).astype(np.uint8)
    randaugment_op = vision.RandAugment()
    with pytest.raises(RuntimeError, match=".*RandAugment: the dimension of image tensor does not match the requirement"
                                           " of operator. Expecting tensor in dimension of \\(3\\), in shape of"
                                           " <H, W, C>. But got dimension 4..*"):
        randaugment_op(image)

    # Test image shape is 0
    image = np.array(10, dtype=np.uint8)
    randaugment_op = vision.RandAugment()
    with pytest.raises(RuntimeError, match=".*RandAugment: the dimension of image tensor does not match the requirement"
                                           " of operator. Expecting tensor in dimension of \\(3\\), in shape of"
                                           " <H, W, C>. But got dimension 0."):
        randaugment_op(image)

    # Test input is null
    randaugment_op = vision.RandAugment()
    with pytest.raises(RuntimeError, match=".*Input Tensor is not valid..*"):
        randaugment_op()

    # Test two input data
    image = np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8)
    randaugment_op = vision.RandAugment()
    with pytest.raises(RuntimeError, match=".*The op is OneToOne, can only accept"
                                           " one tensor as input..*"):
        randaugment_op(image, image)

    # test num_ops is str
    with pytest.raises(TypeError, match="Argument num_ops with value 0 is not of type \\[<class 'int'>\\],"
                                        " but got <class 'str'>."):
        vision.RandAugment(num_ops="0")

    # test num_ops is None
    with pytest.raises(TypeError, match="Argument num_ops with value None is not of type \\[<class 'int'>\\],"
                                        " but got <class 'NoneType'>."):
        vision.RandAugment(num_ops=None)

    # test num_ops is -1
    with pytest.raises(ValueError, match="Input num_ops is not within the required interval of \\[0, 16777216\\]."):
        vision.RandAugment(num_ops=-1)

    # test num_ops is float
    with pytest.raises(TypeError, match="Argument num_ops with value 0.5 is not of type \\[<class 'int'>\\],"
                                        " but got <class 'float'>."):
        vision.RandAugment(num_ops=0.5)

    # test magnitude is str
    with pytest.raises(TypeError, match="Argument magnitude with value 1 is not of type \\[<class 'int'>\\],"
                                        " but got <class 'str'>."):
        vision.RandAugment(magnitude="1")

    # test magnitude is None
    with pytest.raises(TypeError, match="Argument magnitude with value None is not of type \\[<class 'int'>\\],"
                                        " but got <class 'NoneType'>."):
        vision.RandAugment(magnitude=None)

    # test magnitude is -1 ,num_magnitude_bins default 31
    with pytest.raises(ValueError, match="Input magnitude is not within the required interval of \\[0, 31\\)."):
        vision.RandAugment(magnitude=-1)

    # test magnitude is equal num_magnitude_bins
    with pytest.raises(ValueError, match="Input magnitude is not within the required interval of \\[0, 90\\)."):
        vision.RandAugment(magnitude=90, num_magnitude_bins=90)

    # test magnitude is float
    with pytest.raises(TypeError, match="Argument magnitude with value 10.5 is not of type \\[<class 'int'>\\],"
                                        " but got <class 'float'>."):
        vision.RandAugment(magnitude=10.5)

    # test num_magnitude_bins is str
    with pytest.raises(TypeError, match="Argument num_magnitude_bins with value 1 is not of type \\[<class 'int'>\\],"
                                        " but got <class 'str'>."):
        vision.RandAugment(num_magnitude_bins="1")


def test_rand_augment_exception_02():
    """
    Feature: RandAugment operation
    Description: Testing the RandAugment Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # test num_magnitude_bins is None
    with pytest.raises(TypeError, match="Argument num_magnitude_bins with value None is not of type \\[<class 'int'>\\]"
                                        ", but got <class 'NoneType'>."):
        vision.RandAugment(num_magnitude_bins=None)

    # test magnitude is more than num_magnitude_bins, magnitude should be smaller than num_magnitude_bins
    image = np.fromfile(image_file, dtype=np.int32)
    op1 = vision.Decode()
    image_origin = op1(image)
    with pytest.raises(ValueError, match="Input magnitude is not within the required interval of \\[0, 2\\)."):
        op2 = vision.RandAugment(num_ops=13, magnitude=12, num_magnitude_bins=2, interpolation=Inter.BILINEAR,
                                 fill_value=1)
        op2(image_origin)

    # test num_magnitude_bins is less than 2
    with pytest.raises(ValueError, match="Input num_magnitude_bins is not within the required interval of "
                                         "\\[2, 16777216\\]."):
        vision.RandAugment(magnitude=1, num_magnitude_bins=1)

    # test num_magnitude_bins is float
    with pytest.raises(TypeError, match="Argument num_magnitude_bins with value 100.05 is not of type "
                                        "\\[<class 'int'>\\], but got <class 'float'>."):
        vision.RandAugment(num_magnitude_bins=100.05)

    # test interpolation is str
    with pytest.raises(TypeError, match="Argument interpolation with value Inter.AREA is not of type \\[<enum"
                                        " 'Inter'>\\], but got <class 'str'>."):
        vision.RandAugment(interpolation="Inter.AREA")

    # test interpolation is list
    with pytest.raises(TypeError, match="Argument interpolation with value \\[<Inter.BICUBIC: 3>\\] is not of type"
                                        " \\[<enum 'Inter'>\\], but got <class 'list'>."):
        vision.RandAugment(interpolation=[Inter.BICUBIC])

    # test interpolation is not support value Inter.PILCUBIC
    image = Image.open(image_file)
    num_ops = 3
    magnitude = 2
    num_magnitude_bins = 3
    interpolation = Inter.PILCUBIC
    fill_value = (0, 100, 255)
    randaugment_op = vision.RandAugment(num_ops=num_ops, magnitude=magnitude, num_magnitude_bins=num_magnitude_bins,
                                        interpolation=interpolation, fill_value=fill_value)
    with pytest.raises(RuntimeError, match="Invalid InterpolationMode."):
        randaugment_op(image)
    image.close()

    # test interpolation is int
    with pytest.raises(TypeError, match="Argument interpolation with value 0 is not of type"
                                        " \\[<enum 'Inter'>\\], but got <class 'int'>."):
        vision.RandAugment(interpolation=0)

    # test fill_value is str
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.RandAugment(fill_value="0")

    # test fill_value is list
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.RandAugment(fill_value=[0, 1])

    # test fill_value is float
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.RandAugment(fill_value=1.50)

    # test fill_value is -1
    with pytest.raises(ValueError, match="Input fill_value is not within the required interval of \\[0, 255\\]."):
        vision.RandAugment(fill_value=-1)

    # test fill_value isnot 3-tuple
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.RandAugment(fill_value=(1, 2, 3, 4))

    # test fill_value isnot 3-tuple
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.RandAugment(fill_value=(1, 2))

    # test fill_value is 3-float-tuple
    with pytest.raises(TypeError, match="Argument fill_value\\[0\\] with value 1.1 is not of type \\[<class 'int'>\\],"
                                        " but got <class 'float'>."):
        vision.RandAugment(fill_value=(1.1, 2.22, 3.333))

    # test fill_value is 256
    with pytest.raises(ValueError, match="Input fill_value is not within the required interval of \\[0, 255\\]."):
        vision.RandAugment(fill_value=256)

    # test fill_value's 3-tuple is 256
    with pytest.raises(ValueError, match="Input fill_value\\[2\\] is not within the required interval of"
                                         " \\[0, 255\\]."):
        vision.RandAugment(fill_value=(0, 1, 256))


if __name__ == "__main__":
    test_rand_augment_pipeline(plot=True)
    test_rand_augment_eager()
    test_rand_augment_invalid_params_int()
    test_rand_augment_invalid_interpolation()
    test_rand_augment_invalid_fill_value()
    test_rand_augment_invalid_magnitude_value()
    test_rand_augment_operation_01()
    test_rand_augment_exception_01()
    test_rand_augment_exception_02()
