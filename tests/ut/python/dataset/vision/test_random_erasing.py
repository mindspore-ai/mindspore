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
Testing RandomErasing op in DE
"""
import numpy as np
import os
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.transforms.transforms as trans
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger
from util import diff_mse, visualize_image, save_and_check_md5_pil, \
    config_get_set_seed, config_get_set_num_parallel_workers

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
TEST_DATA_DATASET_FUNC ="../data/dataset/"

DATA_DIR_1 = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")

GENERATE_GOLDEN = False


def test_random_erasing_op(plot=False):
    """
    Feature: RandomErasing op
    Description: Test RandomErasing and CutOut
    Expectation: Passes the test
    """
    logger.info("test_random_erasing")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms_1 = [
        vision.Decode(True),
        vision.ToTensor(),
        vision.RandomErasing(value='random')
    ]
    transform_1 = trans.Compose(transforms_1)
    data1 = data1.map(operations=transform_1, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms_2 = [
        vision.Decode(True),
        vision.ToTensor(),
        vision.CutOut(80, is_hwc=False)
    ]
    transform_2 = trans.Compose(transforms_2)
    data2 = data2.map(operations=transform_2, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        num_iter += 1
        image_1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_2 = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)

        logger.info("shape of image_1: {}".format(image_1.shape))
        logger.info("shape of image_2: {}".format(image_2.shape))

        logger.info("dtype of image_1: {}".format(image_1.dtype))
        logger.info("dtype of image_2: {}".format(image_2.dtype))

        mse = diff_mse(image_1, image_2)
        if plot:
            visualize_image(image_1, image_2, mse)


def test_random_erasing_md5():
    """
    Feature: RandomErasing op
    Description: Test RandomErasing with md5 check
    Expectation: Passes the md5 check
    """
    logger.info("Test RandomErasing with md5 check")
    original_seed = config_get_set_seed(5)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms_1 = [
        vision.Decode(True),
        vision.ToTensor(),
        vision.RandomErasing(value='random')
    ]
    transform_1 = trans.Compose(transforms_1)
    data = data.map(operations=transform_1, input_columns=["image"])
    # Compare with expected md5 from images
    filename = "random_erasing_01_result.npz"
    save_and_check_md5_pil(data, filename, generate_golden=GENERATE_GOLDEN)

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers((original_num_parallel_workers))


def test_random_erasing_operation_01():
    """
    Feature: RandomErasing operation
    Description: Testing the normal functionality of the RandomErasing operator
    Expectation: The Output is equal to the expected output
    """
    # When parameter value is a 3-tuple, the RandomErasing interface is successfully called
    prob = 0.5
    scale = (0.02, 0.33)
    ratio = (0.3, 3.3)
    value = (5, 1, 5)
    inplace = False
    max_attempts = 10
    dataset = ds.ImageFolderDataset(DATA_DIR_1, 1)
    transforms1 = [
        vision.Decode(),
        vision.ToTensor(),
        vision.RandomErasing(prob, scale, ratio, value, inplace, max_attempts),
    ]
    transform1 = trans.Compose(transforms1)
    dataset = dataset.map(input_columns=["image"], operations=transform1)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When parameter prob is 0, the RandomErasing interface is successfully called
    prob = 0
    scale = (0.02, 0.33)
    ratio = (0.3, 3.3)
    value = (5, 5, 5)
    inplace = False
    max_attempts = 10
    dataset = ds.ImageFolderDataset(DATA_DIR_1, 1)
    transforms1 = [
        vision.Decode(),
        vision.ToTensor(),
        vision.RandomErasing(prob, scale, ratio, value, inplace, max_attempts),
    ]
    transform1 = trans.Compose(transforms1)
    dataset = dataset.map(input_columns=["image"], operations=transform1)

    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When parameter prob is 1, the RandomErasing interface is successfully called
    prob = 1
    scale = (0.02, 0.33)
    ratio = (0.3, 3.3)
    value = 128
    inplace = False
    max_attempts = 10
    dataset = ds.ImageFolderDataset(DATA_DIR_1, 1)
    transforms1 = [
        vision.Decode(),
        vision.ToTensor(),
        vision.RandomErasing(prob, scale, ratio, value, inplace, max_attempts),
    ]
    transform1 = trans.Compose(transforms1)
    dataset = dataset.map(input_columns=["image"], operations=transform1)

    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass


def test_random_erasing_operation_02():
    """
    Feature: RandomErasing operation
    Description: Testing the normal functionality of the RandomErasing operator
    Expectation: The Output is equal to the expected output
    """
    # When parameter scale is (3, 10), the RandomErasing interface is successfully called
    prob = 1
    scale = (3, 10)
    ratio = (0.3, 3.3)
    value = [20, 1, 255]
    inplace = False
    max_attempts = 10
    dataset = ds.ImageFolderDataset(DATA_DIR_1, 1)
    transforms1 = [
        vision.Decode(),
        vision.ToTensor(),
        vision.RandomErasing(prob, scale, ratio, value, inplace, max_attempts),
    ]
    transform1 = trans.Compose(transforms1)
    dataset = dataset.map(input_columns=["image"], operations=transform1)

    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # When parameter ratio is (3, 33), the RandomErasing interface is successfully called
    prob = 0.5
    scale = (0.02, 0.33)
    ratio = (3, 33)
    value = (5, 1, 150)
    inplace = True
    max_attempts = 10
    dataset = ds.ImageFolderDataset(DATA_DIR_1, 1)
    transforms1 = [
        vision.Decode(),
        vision.ToTensor(),
        vision.RandomErasing(prob, scale, ratio, value, inplace, max_attempts),
    ]
    transform1 = trans.Compose(transforms1)
    dataset = dataset.map(input_columns=["image"], operations=transform1)

    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Using default parameter values, the RandomErasing interface is successfully called
    dataset = ds.ImageFolderDataset(DATA_DIR_1, 1)
    transforms1 = [
        vision.Decode(),
        vision.ToTensor(),
        vision.RandomErasing()
    ]
    transform1 = trans.Compose(transforms1)
    dataset = dataset.map(input_columns=["image"], operations=transform1)

    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass


def test_random_erasing_operation_03():
    """
    Feature: RandomErasing operation
    Description: Testing the normal functionality of the RandomErasing operator
    Expectation: The Output is equal to the expected output
    """
    # When the input image format is jpg, the RandomErasing interface is successfully called
    with Image.open(image_jpg) as image1:
        image = vision.ToTensor()(image1)
        prob = 1.0
        scale = (0.02, 0.33)
        ratio = (0.3, 5.3)
        value = (100, 80, 60)
        inplace = False
        max_attempts = 10
        random_erasing_op = vision.RandomErasing(prob, scale, ratio, value, inplace, max_attempts)
        _ = random_erasing_op(image)

    # When the input image format is bmp, the RandomErasing interface is successfully called
    with Image.open(image_bmp) as image1:
        image = vision.ToTensor()(image1)
        prob = 0.985
        scale = [0.9, 0.9]
        ratio = (0.5, 0.8)
        value = 255
        inplace = True
        max_attempts = 1
        random_erasing_op = vision.RandomErasing(prob, scale, ratio, value, inplace, max_attempts)
        _ = random_erasing_op(image)

    # When the input image format is gif, the RandomErasing interface is successfully called
    with Image.open(image_gif) as image1:
        image = vision.ToTensor()(image1)
        prob = 0.5
        scale = (0, 0.6)
        ratio = [1, 1]
        value = 5
        inplace = False
        max_attempts = 3
        random_erasing_op = vision.RandomErasing(prob, scale, ratio, value, inplace, max_attempts)
        _ = random_erasing_op(image)

    # When the input image format is png, the RandomErasing interface is successfully called
    with Image.open(image_png) as image1:
        image = vision.ToTensor()(image1)
        prob = 1
        scale = [0.5, 0.5]
        ratio = [0.3, 1.6]
        value = "random"
        inplace = True
        max_attempts = 20
        random_erasing_op = vision.RandomErasing(prob, scale, ratio, value, inplace, max_attempts)
        _ = random_erasing_op(image)

    # When the input data is numpy, the RandomErasing interface is successfully called
    image1 = np.random.randint(0, 255, (300, 400, 3)).astype(np.uint8)
    image = np.transpose(image1, (2, 0, 1))
    prob = 0.8
    scale = (0.3, 0.8)
    ratio = (0.5, 0.6)
    value = [5, 128, 58]
    inplace = True
    max_attempts = 10
    random_erasing_op = vision.RandomErasing(prob, scale, ratio, value, inplace, max_attempts)
    _ = random_erasing_op(image)

    # In eager mode, using default parameter values, the RandomErasing interface is successfully called
    with Image.open(image_jpg) as image1:
        image = vision.ToTensor()(image1)
        random_erasing_op = vision.RandomErasing()
        _ = random_erasing_op(image)

    # In eager mode, when parameter prob is 0, the RandomErasing interface is successfully called
    with Image.open(image_png) as image1:
        image = vision.ToTensor()(image1)
        prob = 0
        random_erasing_op = vision.RandomErasing(prob)
        _ = random_erasing_op(image)


def test_random_erasing_exception_01():
    """
    Feature: RandomErasing operation
    Description: Testing the RandomErasing Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When parameter max_attempts is 0, the RandomErasing interface call fails
    prob = 0.5
    scale = (0.02, 0.33)
    ratio = (0.3, 3.3)
    value = 0
    inplace = False
    max_attempts = 0
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        vision.RandomErasing(prob, scale, ratio, value, inplace, max_attempts)

    # When parameter max_attempts is negative, the RandomErasing interface call fails
    prob = 0.5
    scale = (0.02, 0.33)
    ratio = (0.3, 3.3)
    value = 0
    inplace = False
    max_attempts = -1
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        vision.RandomErasing(prob, scale, ratio, value, inplace, max_attempts)

    # When parameter value is of float type, the RandomErasing interface call fails
    prob = 0.5
    scale = (0.02, 0.33)
    ratio = (0.3, 3.3)
    value = (0.5, 0.1)
    inplace = False
    max_attempts = 10
    with pytest.raises(ValueError, match=("should be either a single value, or a string 'random',"
                                          " or a sequence of 3 elements for RGB respectively")):
        vision.RandomErasing(prob, scale, ratio, value, inplace, max_attempts)

    # When the input shape is 4-dimensional, the RandomErasing interface call fails
    image = np.random.randint(0, 255, (300, 400, 3, 3)).astype(np.uint8)
    prob = 1
    random_erasing_op = vision.RandomErasing(prob)
    with pytest.raises(ValueError, match="too many values to unpack \\(expected 3\\)"):
        random_erasing_op(image)

    # When the input data is a list, the RandomErasing interface call fails
    image = np.random.randint(0, 255, (300, 400, 3)).astype(np.uint8).tolist()
    prob = 1
    random_erasing_op = vision.RandomErasing(prob)
    with pytest.raises(TypeError, match="img should be NumPy array. Got <class 'list'>."):
        random_erasing_op(image)

    # When the input data is a tuple, the RandomErasing interface call fails
    image = tuple(np.random.randint(0, 255, (300, 400, 3)).astype(np.uint8))
    prob = 1
    random_erasing_op = vision.RandomErasing(prob)
    with pytest.raises(TypeError, match="img should be NumPy array. Got <class 'tuple'>."):
        random_erasing_op(image)

    # When the input is PIL-PngImageFile, the RandomErasing interface call fails
    with Image.open(image_png) as image:
        prob = 1
        random_erasing_op = vision.RandomErasing(prob)
        with pytest.raises(TypeError, match="img should be NumPy array. "
                                            "Got <class 'PIL.PngImagePlugin.PngImageFile'>."):
            random_erasing_op(image)

    # When parameter prob is negative, the RandomErasing interface call fails
    prob = -0.1
    with pytest.raises(ValueError, match="Input prob is not within the required interval of \\[0.0, 1.0\\]."):
        vision.RandomErasing(prob)

    # When parameter prob is greater than 1.0, the RandomErasing interface call fails
    prob = 1.1
    with pytest.raises(ValueError, match="Input prob is not within the required interval of \\[0.0, 1.0\\]."):
        vision.RandomErasing(prob)

    # When parameter prob is a string, the RandomErasing interface call fails
    prob = "1"
    with pytest.raises(TypeError,
                       match="Argument prob with value 1 is not of type \\[<class 'float'>, <class 'int'>\\]."):
        vision.RandomErasing(prob=prob)

    # When parameter prob is a list, the RandomErasing interface call fails
    prob = [1]
    with pytest.raises(TypeError, match="Argument prob with value \\[1\\] is not of "
                                        "type \\[<class 'float'>, <class 'int'>\\]."):
        vision.RandomErasing(prob=prob)

    # When parameter scale is negative, the RandomErasing interface call fails
    scale = (-0.1, 1)
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[0, 16777216\\]."):
        vision.RandomErasing(scale=scale)

    # When parameter scale is greater than 16777216, the RandomErasing interface call fails
    scale = (0.1, 16777216.1)
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[0, 16777216\\]."):
        vision.RandomErasing(scale=scale)

    # When the length of parameter scale is less than 2, the RandomErasing interface call fails
    scale = (0.1,)
    with pytest.raises(TypeError, match="scale should be a list or tuple of length 2."):
        vision.RandomErasing(scale=scale)


def test_random_erasing_exception_02():
    """
    Feature: RandomErasing operation
    Description: Testing the RandomErasing Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When the length of parameter scale is greater than 3, the RandomErasing interface call fails
    scale = (0.1, 0.2, 0.3)
    with pytest.raises(TypeError, match="scale should be a list or tuple of length 2."):
        vision.RandomErasing(scale=scale)

    # When the first value in parameter scale is greater than the second value, the RandomErasing interface call fails
    scale = (0.2, 0.1)
    with pytest.raises(ValueError, match="scale should be in \\(min,max\\) format. Got \\(max,min\\)."):
        vision.RandomErasing(scale=scale)

    # When parameter scale is a numpy array, the RandomErasing interface call fails
    scale = np.array([0.1, 0.2])
    with pytest.raises(TypeError, match="Argument scale with value \\[0.1 0.2\\] is not of "
                                        "type \\[<class 'list'>, <class 'tuple'>\\]."):
        vision.RandomErasing(scale=scale)

    # When parameter scale is a set, the RandomErasing interface call fails
    scale = {0.1, 0.2}
    with pytest.raises(TypeError, match="Argument scale with value {0.1, 0.2} is not of "
                                        "type \\[<class 'list'>, <class 'tuple'>\\]."):
        vision.RandomErasing(scale=scale)

    # When parameter scale value is a single value, the RandomErasing interface call fails
    scale = 0.3
    with pytest.raises(TypeError, match="Argument scale with value 0.3 is not of "
                                        "type \\[<class 'list'>, <class 'tuple'>\\]."):
        vision.RandomErasing(scale=scale)

    # When parameter ratio is 0, the RandomErasing interface call fails
    prob = 0.6
    scale = (0.1, 0.2)
    ratio = (0, 1)
    with pytest.raises(ValueError, match="Input is not within the required interval of \\(0, 16777216\\]."):
        vision.RandomErasing(prob=prob, scale=scale, ratio=ratio)

    # When parameter ratio is greater than 16777216, the RandomErasing interface call fails
    prob = 0.6
    scale = (0.1, 0.2)
    ratio = (1, 16777216.1)
    with pytest.raises(ValueError, match="Input is not within the required interval of \\(0, 16777216\\]."):
        vision.RandomErasing(prob=prob, scale=scale, ratio=ratio)

    # When parameter ratio is negative, the RandomErasing interface call fails
    prob = 0.6
    scale = (0.1, 0.2)
    ratio = (-0.1, 1)
    with pytest.raises(ValueError, match="Input is not within the required interval of \\(0, 16777216\\]."):
        vision.RandomErasing(prob=prob, scale=scale, ratio=ratio)

    # When the length of parameter ratio is less than 2, the RandomErasing interface call fails
    prob = 0.6
    scale = (0.1, 0.2)
    ratio = [0.2]
    with pytest.raises(TypeError, match="ratio should be a list or tuple of length 2."):
        vision.RandomErasing(prob=prob, scale=scale, ratio=ratio)

    # When the length of parameter ratio is greater than 2, the RandomErasing interface call fails
    prob = 0.6
    scale = (0.1, 0.2)
    ratio = [0.2, 0.3, 0.4]
    with pytest.raises(TypeError, match="ratio should be a list or tuple of length 2."):
        vision.RandomErasing(prob=prob, scale=scale, ratio=ratio)

    # When the second value of parameter ratio is less than the first value, the RandomErasing interface call fails
    prob = 0.6
    scale = (0.1, 0.2)
    ratio = [0.3, 0.2]
    with pytest.raises(ValueError, match="ratio should be in \\(min,max\\) format. Got \\(max,min\\)."):
        vision.RandomErasing(prob=prob, scale=scale, ratio=ratio)

    # When parameter ratio is a set, the RandomErasing interface call fails
    prob = 0.6
    scale = (0.1, 0.2)
    ratio = {0.2, 0.3}
    with pytest.raises(TypeError, match="Argument ratio with value {0.2, 0.3} is not of "
                                        "type \\[<class 'list'>, <class 'tuple'>\\]."):
        vision.RandomErasing(prob=prob, scale=scale, ratio=ratio)

    # When parameter ratio is a numpy array, the RandomErasing interface call fails
    prob = 0.6
    scale = (0.1, 0.2)
    ratio = np.array([0.2, 0.3])
    with pytest.raises(TypeError, match="Argument ratio with value \\[0.2 0.3\\] is not of type \\[<class 'list'>, "
                                        "<class 'tuple'>\\]."):
        vision.RandomErasing(prob=prob, scale=scale, ratio=ratio)

    # When parameter ratio is 1, the RandomErasing interface call fails
    prob = 0.6
    scale = (0.1, 0.2)
    ratio = 1
    with pytest.raises(TypeError,
                       match="Argument ratio with value 1 is not of type \\[<class 'list'>, <class 'tuple'>\\]."):
        vision.RandomErasing(prob=prob, scale=scale, ratio=ratio)

    # When parameter value is an array of length 2, the RandomErasing interface call fails
    prob = 0.6
    value = [25, 100]
    with pytest.raises(ValueError, match="The value for erasing should be either a single value, or a string"
                                         " 'random', or a sequence of 3 elements for RGB respectively."):
        vision.RandomErasing(prob=prob, value=value)


def test_random_erasing_exception_03():
    """
    Feature: RandomErasing operation
    Description: Testing the RandomErasing Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When parameter value is an array of length 4, the RandomErasing interface call fails
    prob = 0.6
    value = [25, 100, 80, 50]
    with pytest.raises(ValueError, match="The value for erasing should be either a single value, or a string"
                                         " 'random', or a sequence of 3 elements for RGB respectively."):
        vision.RandomErasing(prob=prob, value=value)

    # When parameter value is a float, the RandomErasing interface call fails
    prob = 0.6
    value = 50.0
    with pytest.raises(TypeError, match="Argument value with value 50.0 is not of type \\[<class 'int'>, <class"
                                        " 'list'>, <class 'tuple'>, <class 'str'>\\]."):
        vision.RandomErasing(prob=prob, value=value)

    # When parameter value is greater than 255, the RandomErasing interface call fails
    prob = 0.6
    value = (50, 80, 256)
    with pytest.raises(ValueError, match="Input value is not within the required interval of \\[0, 255\\]."):
        vision.RandomErasing(prob=prob, value=value)

    # When parameter value is negative, the RandomErasing interface call fails
    prob = 0.6
    value = -1
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[0, 255\\]."):
        vision.RandomErasing(prob=prob, value=value)

    # When parameter value is a set, the RandomErasing interface call fails
    value = {25, 100, 80}
    with pytest.raises(TypeError, match="Argument value with value {80, 25, 100} is not of type \\[<class 'int'>, "
                                        "<class 'list'>, <class 'tuple'>, <class 'str'>\\]."):
        vision.RandomErasing(value=value)

    # When parameter value is a numpy array, the RandomErasing interface call fails
    value = np.array([25, 100, 80])
    with pytest.raises(TypeError, match="Argument value with value \\[ 25 100  80\\] is not of type \\[<class 'int'>, "
                                        "<class 'list'>, <class 'tuple'>, <class 'str'>\\]."):
        vision.RandomErasing(value=value)

    # When parameter inplace is an int, the RandomErasing interface call fails
    value = [25, 100, 80]
    inplace = 1
    with pytest.raises(TypeError, match="Argument inplace with value 1 is not of type \\[<class 'bool'>\\]."):
        vision.RandomErasing(value=value, inplace=inplace)

    # When parameter inplace is a string, the RandomErasing interface call fails
    value = [25, 100, 80]
    inplace = "True"
    with pytest.raises(TypeError, match="Argument inplace with value True is not of type \\[<class 'bool'>\\]"):
        vision.RandomErasing(value=value, inplace=inplace)

    # When parameter inplace is a list, the RandomErasing interface call fails
    inplace = [True]
    with pytest.raises(TypeError, match="Argument inplace with value \\[True\\] is not of type \\[<class 'bool'>\\]."):
        vision.RandomErasing(inplace=inplace)

    # When parameter max_attempts is greater than 16777216, the RandomErasing interface call fails
    max_attempts = 16777217
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[1, 16777216\\]."):
        vision.RandomErasing(max_attempts=max_attempts)

    # When parameter max_attempts is a float, the RandomErasing interface call fails
    max_attempts = 10.0
    with pytest.raises(TypeError, match="Argument max_attempts with value 10.0 is not of type \\[<class 'int'>\\]."):
        vision.RandomErasing(max_attempts=max_attempts)

    # When parameter max_attempts is of bool type, the RandomErasing interface call fails
    max_attempts = True
    with pytest.raises(TypeError, match="Argument max_attempts with value True is not of type \\(<class 'int'>,\\)."):
        vision.RandomErasing(max_attempts=max_attempts)

    # When parameter max_attempts is a list, the RandomErasing interface call fails
    max_attempts = [5]
    with pytest.raises(TypeError,
                       match="Argument max_attempts with value \\[5\\] is not of type \\[<class 'int'>\\]."):
        vision.RandomErasing(max_attempts=max_attempts)

    # When parameter max_attempts is a string, the RandomErasing interface call fails
    max_attempts = "5"
    with pytest.raises(TypeError, match="Argument max_attempts with value 5 is not of type \\[<class 'int'>\\]."):
        vision.RandomErasing(max_attempts=max_attempts)


if __name__ == "__main__":
    test_random_erasing_op(plot=True)
    test_random_erasing_md5()
    test_random_erasing_operation_01()
    test_random_erasing_operation_02()
    test_random_erasing_operation_03()
    test_random_erasing_exception_01()
    test_random_erasing_exception_02()
    test_random_erasing_exception_03()
