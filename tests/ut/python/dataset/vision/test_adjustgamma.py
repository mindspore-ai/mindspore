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
Testing AdjustGamma op in DE
"""
import cv2
import numpy as np
from numpy.testing import assert_allclose
from PIL import Image
import os
import pytest

import mindspore.dataset as ds
import mindspore.dataset.transforms
import mindspore.dataset.transforms.transforms as t_trans
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger

DATA_DIR = "../data/dataset/testImageNetData/train/"

DATA_DIR_2 = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

TEST_DATA_DATASET_FUNC = "../data/dataset/"


def dir_data():
    """Obtain the dataset"""
    data_list = []
    data_dir1 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "testImageNetData", "train")
    data_dir2 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    data_dir3 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")
    data_dir4 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
    data_dir5 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    data_list.append(data_dir1)
    data_list.append(data_dir2)
    data_list.append(data_dir3)
    data_list.append(data_dir4)
    data_list.append(data_dir5)
    return data_list


def generate_numpy_random_rgb(shape):
    """
    Only generate floating points that are fractions like n / 256, since they
    are RGB pixels. Some low-precision floating point types in this test can't
    handle arbitrary precision floating points well.
    """
    return np.random.randint(0, 256, shape) / 255.


def test_adjust_gamma_c_eager():
    """
    Feature: AdjustGamma op
    Description: Test eager support for AdjustGamma Cpp implementation
    Expectation: Receive non-None output image from op
    """
    # Eager 3-channel
    rgb_flat = generate_numpy_random_rgb((64, 3)).astype(np.float32)
    img_in = rgb_flat.reshape((8, 8, 3))

    adjustgamma_op = vision.AdjustGamma(10, 1)
    img_out = adjustgamma_op(img_in)
    assert img_out is not None

    img_in2 = Image.open("../data/dataset/apple.jpg").convert("RGB")

    adjustgamma_op2 = vision.AdjustGamma(10, 1)
    img_out2 = adjustgamma_op2(img_in2)
    assert img_out2 is not None


def test_adjust_gamma_py_eager():
    """
    Feature: AdjustGamma op
    Description: Test eager support for AdjustGamma Python implementation
    Expectation: Receive non-None output image from op
    """
    # Eager 3-channel
    rgb_flat = generate_numpy_random_rgb((64, 3)).astype(np.uint8)
    img_in = Image.fromarray(rgb_flat.reshape((8, 8, 3)))

    adjustgamma_op = vision.AdjustGamma(10, 1)
    img_out = adjustgamma_op(img_in)
    assert img_out is not None

    img_in2 = Image.open("../data/dataset/apple.jpg").convert("RGB")

    adjustgamma_op2 = vision.AdjustGamma(10, 1)
    img_out2 = adjustgamma_op2(img_in2)
    assert img_out2 is not None


def test_adjust_gamma_c_eager_gray():
    """
    Feature: AdjustGamma op
    Description: Test eager support for AdjustGamma Cpp implementation 1-channel
    Expectation: Receive non-None output image from op
    """
    # Eager 1-channel
    rgb_flat = generate_numpy_random_rgb((64, 1)).astype(np.float32)
    img_in = rgb_flat.reshape((8, 8))

    adjustgamma_op = vision.AdjustGamma(10, 1)
    img_out = adjustgamma_op(img_in)
    assert img_out is not None


def test_adjust_gamma_py_eager_gray():
    """
    Feature: AdjustGamma op
    Description: Test eager support for AdjustGamma Python implementation 1-channel
    Expectation: Receive non-None output image from op
    """
    # Eager 1-channel
    rgb_flat = generate_numpy_random_rgb((64, 1)).astype(np.uint8)
    img_in = Image.fromarray(rgb_flat.reshape((8, 8)))

    adjustgamma_op = vision.AdjustGamma(10, 1)
    img_out = adjustgamma_op(img_in)
    assert img_out is not None


def test_adjust_gamma_invalid_gamma_param_c():
    """
    Feature: AdjustGamma op
    Description: Test AdjustGamma Cpp implementation with invalid ignore parameter
    Expectation: Correct error is raised as expected
    """
    logger.info(
        "Test AdjustGamma C implementation with invalid ignore parameter")
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(
            operations=[vision.Decode(), vision.Resize((224, 224)),
                        lambda img: np.array(img[:, :, 0])],
            input_columns=["image"])
        # invalid gamma
        data_set = data_set.map(operations=vision.AdjustGamma(gamma=-10.0, gain=1.0),
                                input_columns="image")
    except ValueError as error:
        logger.info("Got an exception in AdjustGamma: {}".format(str(error)))
        assert "Input is not within the required interval of " in str(error)
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(
            operations=[vision.Decode(), vision.Resize((224, 224)),
                        lambda img: np.array(img[:, :, 0])],
            input_columns=["image"])
        # invalid gamma
        data_set = data_set.map(operations=vision.AdjustGamma(gamma=[1, 2], gain=1.0),
                                input_columns="image")
    except TypeError as error:
        logger.info("Got an exception in AdjustGamma: {}".format(str(error)))
        assert "is not of type [<class 'float'>, <class 'int'>], but got" in str(
            error)


def test_adjust_gamma_invalid_gamma_param_py():
    """
    Feature: AdjustGamma op
    Description: Test AdjustGamma Python implementation with invalid ignore parameter
    Expectation: Correct error is raised as expected
    """
    logger.info(
        "Test AdjustGamma Python implementation with invalid ignore parameter")
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        trans = mindspore.dataset.transforms.Compose([
            vision.Decode(True),
            vision.Resize((224, 224)),
            vision.AdjustGamma(gamma=-10.0),
            vision.ToTensor()
        ])
        data_set = data_set.map(operations=[trans], input_columns=["image"])
    except ValueError as error:
        logger.info("Got an exception in AdjustGamma: {}".format(str(error)))
        assert "Input is not within the required interval of " in str(error)
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        trans = mindspore.dataset.transforms.Compose([
            vision.Decode(True),
            vision.Resize((224, 224)),
            vision.AdjustGamma(gamma=[1, 2]),
            vision.ToTensor()
        ])
        data_set = data_set.map(operations=[trans], input_columns=["image"])
    except TypeError as error:
        logger.info("Got an exception in AdjustGamma: {}".format(str(error)))
        assert "is not of type [<class 'float'>, <class 'int'>], but got" in str(
            error)


def test_adjust_gamma_invalid_gain_param_c():
    """
    Feature: AdjustGamma op
    Description: Test AdjustGamma Cpp implementation with invalid gain parameter
    Expectation: Correct error is raised as expected
    """
    logger.info("Test AdjustGamma C implementation with invalid gain parameter")
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(
            operations=[vision.Decode(), vision.Resize((224, 224)),
                        lambda img: np.array(img[:, :, 0])],
            input_columns=["image"])
        # invalid gain
        data_set = data_set.map(operations=vision.AdjustGamma(gamma=10.0, gain=[1, 10]),
                                input_columns="image")
    except TypeError as error:
        logger.info("Got an exception in AdjustGamma: {}".format(str(error)))
        assert "is not of type [<class 'float'>, <class 'int'>], but got " in str(
            error)


def test_adjust_gamma_invalid_gain_param_py():
    """
    Feature: AdjustGamma op
    Description: Test AdjustGamma Python implementation with invalid gain parameter
    Expectation: Correct error is raised as expected
    """
    logger.info(
        "Test AdjustGamma Python implementation with invalid gain parameter")
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        trans = mindspore.dataset.transforms.Compose([
            vision.Decode(True),
            vision.Resize((224, 224)),
            vision.AdjustGamma(gamma=10.0, gain=[1, 10]),
            vision.ToTensor()
        ])
        data_set = data_set.map(operations=[trans], input_columns=["image"])
    except TypeError as error:
        logger.info("Got an exception in AdjustGamma: {}".format(str(error)))
        assert "is not of type [<class 'float'>, <class 'int'>], but got " in str(
            error)


def test_adjust_gamma_pipeline_c():
    """
    Feature: AdjustGamma op
    Description: Test AdjustGamma Cpp implementation Pipeline
    Expectation: Runs successfully
    """
    # First dataset
    transforms1 = [vision.Decode(), vision.Resize([64, 64])]
    transforms1 = mindspore.dataset.transforms.Compose(
        transforms1)
    ds1 = ds.TFRecordDataset(DATA_DIR_2,
                             SCHEMA_DIR,
                             columns_list=["image"],
                             shuffle=False)
    ds1 = ds1.map(operations=transforms1, input_columns=["image"])

    # Second dataset
    transforms2 = [
        vision.Decode(),
        vision.Resize([64, 64]),
        vision.AdjustGamma(1.0, 1.0)
    ]
    transform2 = mindspore.dataset.transforms.Compose(
        transforms2)
    ds2 = ds.TFRecordDataset(DATA_DIR_2,
                             SCHEMA_DIR,
                             columns_list=["image"],
                             shuffle=False)
    ds2 = ds2.map(operations=transform2, input_columns=["image"])

    num_iter = 0
    for data1, data2 in zip(ds1.create_dict_iterator(num_epochs=1),
                            ds2.create_dict_iterator(num_epochs=1)):
        num_iter += 1
        ori_img = data1["image"].asnumpy()
        cvt_img = data2["image"].asnumpy()
        assert_allclose(ori_img.flatten(),
                        cvt_img.flatten(),
                        rtol=1e-5,
                        atol=0)
        assert ori_img.shape == cvt_img.shape


def test_adjust_gamma_pipeline_py():
    """
    Feature: AdjustGamma op
    Description: Test AdjustGamma Python implementation Pipeline
    Expectation: Runs successfully
    """
    # First dataset
    transforms1 = [vision.Decode(True), vision.Resize(
        [64, 64]), vision.ToTensor()]
    transforms1 = mindspore.dataset.transforms.Compose(
        transforms1)
    ds1 = ds.TFRecordDataset(DATA_DIR_2,
                             SCHEMA_DIR,
                             columns_list=["image"],
                             shuffle=False)
    ds1 = ds1.map(operations=transforms1, input_columns=["image"])

    # Second dataset
    transforms2 = [
        vision.Decode(True),
        vision.Resize([64, 64]),
        vision.AdjustGamma(1.0, 1.0),
        vision.ToTensor()
    ]
    transform2 = mindspore.dataset.transforms.Compose(
        transforms2)
    ds2 = ds.TFRecordDataset(DATA_DIR_2,
                             SCHEMA_DIR,
                             columns_list=["image"],
                             shuffle=False)
    ds2 = ds2.map(operations=transform2, input_columns=["image"])

    num_iter = 0
    for data1, data2 in zip(ds1.create_dict_iterator(num_epochs=1),
                            ds2.create_dict_iterator(num_epochs=1)):
        num_iter += 1
        ori_img = data1["image"].asnumpy()
        cvt_img = data2["image"].asnumpy()
        assert_allclose(ori_img.flatten(),
                        cvt_img.flatten(),
                        rtol=1e-5,
                        atol=0)
        assert ori_img.shape == cvt_img.shape


def test_adjust_gamma_pipeline_py_gray():
    """
    Feature: AdjustGamma op
    Description: Test AdjustGamma Python implementation Pipeline 1-channel
    Expectation: Runs successfully
    """
    # First dataset
    transforms1_list = [vision.Decode(True), vision.Resize(
        [60, 60]), vision.Grayscale(), vision.ToTensor()]
    transforms1 = mindspore.dataset.transforms.Compose(
        transforms1_list)
    ds1 = ds.TFRecordDataset(DATA_DIR_2,
                             SCHEMA_DIR,
                             columns_list=["image"],
                             shuffle=False)
    ds1 = ds1.map(operations=transforms1, input_columns=["image"])

    # Second dataset
    transforms2_list = [
        vision.Decode(True),
        vision.Resize([60, 60]),
        vision.Grayscale(),
        vision.AdjustGamma(1.0, 1.0),
        vision.ToTensor()
    ]
    transform2 = mindspore.dataset.transforms.Compose(
        transforms2_list)
    ds2 = ds.TFRecordDataset(DATA_DIR_2,
                             SCHEMA_DIR,
                             columns_list=["image"],
                             shuffle=False)
    ds2 = ds2.map(operations=transform2, input_columns=["image"])

    num_iter = 0
    for data1, data2 in zip(ds1.create_dict_iterator(num_epochs=1),
                            ds2.create_dict_iterator(num_epochs=1)):
        num_iter += 1
        ori_img = data1["image"].asnumpy()
        cvt_img = data2["image"].asnumpy()
        assert_allclose(ori_img.flatten(),
                        cvt_img.flatten(),
                        rtol=1e-5,
                        atol=0)


def test_adjust_gamma_eager_image_type():
    """
    Feature: AdjustGamma op
    Description: Test AdjustGamma op eager support test for variety of image input types
    Expectation: Receive non-None output image from op
    """

    def test_config(my_input):
        my_output = vision.AdjustGamma(gamma=1.2, gain=1.0)(my_input)
        assert my_output is not None

    # Test with OpenCV images
    img = cv2.imread("../data/dataset/apple.jpg")
    test_config(img)

    # Test with NumPy array input
    img = np.random.randint(0, 1, (100, 100, 3)).astype(np.uint8)
    test_config(img)

    # Test with PIL Image
    img = Image.open("../data/dataset/apple.jpg").convert("RGB")
    test_config(img)


def test_adjust_gamma_eager_invalid_image_types1():
    """
    Feature: AdjustGamma op
    Description: Exception eager support test for error input type
    Expectation: Error input image is detected
    """

    def test_config(my_input, error_msg):
        with pytest.raises(TypeError) as error_info:
            _ = vision.AdjustGamma(gamma=1.2, gain=1.0)(my_input)
        assert error_msg in str(error_info.value)

    img = Image.open("../data/dataset/apple.jpg").convert("RGB")
    test_config([img, img], "Input should be NumPy or PIL image, got <class 'list'>")
    test_config((img, img), "Input should be NumPy or PIL image, got <class 'tuple'>")

    img = cv2.imread("../data/dataset/apple.jpg")
    test_config([img, img], "Input should be NumPy or PIL image, got <class 'list'>")
    test_config((img, img), "Input should be NumPy or PIL image, got <class 'tuple'>")


def test_adjust_gamma_eager_invalid_image_types2():
    """
    Feature: AdjustGamma op
    Description: Exception eager support test for error input type
    Expectation: Error input image is detected
    """

    def test_config(my_input, error_msg):
        with pytest.raises(TypeError) as error_info:
            _ = vision.AdjustGamma(gamma=1.2, gain=1.0)(my_input)
        assert error_msg in str(error_info.value)

    test_config(1, "Input should be NumPy or PIL image, got <class 'int'>")
    test_config(1.0, "Input should be NumPy or PIL image, got <class 'float'>")
    test_config((1.0, 2.0), "Input should be NumPy or PIL image, got <class 'tuple'>")


def test_adjust_gamma_operation_01():
    """
    Feature: AdjustGamma operation
    Description: Testing the normal functionality of the AdjustGamma operator
    Expectation: The Output is equal to the expected output
    """
    # AdjustGamma Operator: Test Pipeline Mode
    dataset1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    adjust_gamma_op = [vision.Decode(to_pil=False), vision.AdjustGamma(gamma=1)]
    dataset2 = dataset2.map(input_columns=["image"], operations=adjust_gamma_op)
    for _ in zip(dataset1.create_dict_iterator(output_numpy=True), dataset2.create_dict_iterator(output_numpy=True)):
        pass

    # AdjustGamma Operator: Input data is in JPG format
    with Image.open(dir_data()[1]) as image:
        adjust_gamma_op = vision.AdjustGamma(gamma=0, gain=0)
        _ = adjust_gamma_op(image)

    # AdjustGamma Operator: Input data is in GIF format
    with Image.open(dir_data()[2]) as image:
        adjust_gamma_op = vision.AdjustGamma(gamma=30, gain=-125)
        _ = adjust_gamma_op(image)

    # AdjustGamma Operator: Input data is 2D numpy-type data.
    image = np.random.randint(0, 255, (30, 30)).astype(np.uint8)
    adjust_gamma_op = vision.AdjustGamma(gamma=16777216, gain=16777216)
    _ = adjust_gamma_op(image)

    # AdjustGamma Operator: Input data is 3D numpy-type data.
    image = np.random.randn(30, 30, 3)
    adjust_gamma_op = vision.AdjustGamma(gamma=20.0, gain=-16777216)
    _ = adjust_gamma_op(image)

    # AdjustGamma Operator: Input data is 4D numpy-type data.
    image = np.random.randn(20, 10, 15, 1)
    adjust_gamma_op = vision.AdjustGamma(gamma=2.68, gain=0.25)
    _ = adjust_gamma_op(image)

    # AdjustGamma Operator: Input data is 5D numpy-type data.
    image = np.random.randn(8, 12, 10, 5, 3)
    adjust_gamma_op = vision.AdjustGamma(gamma=0.5, gain=-3.6)
    _ = adjust_gamma_op(image)

    # AdjustGamma Operator: gamma parameter set to 1
    ds1 = ds.ImageFolderDataset(dir_data()[0], 1)
    transforms = [vision.Decode(to_pil=True), vision.ToTensor()]
    transform = t_trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ds2 = ds.ImageFolderDataset(dir_data()[0], 1)
    transforms1 = [vision.Decode(to_pil=True), vision.AdjustGamma(gamma=1), vision.ToTensor()]
    transform1 = t_trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)
    for _ in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        pass

    # The AdjustGamma operator: gamma parameter set to 1, gain parameter set to 1677.
    image = np.random.randint(0, 255, (30, 30, 3)).astype(np.uint8)
    to_pil_op = vision.ToPIL()
    image = to_pil_op(image)
    adjust_gamma_op = vision.AdjustGamma(gamma=1677, gain=1677)
    _ = adjust_gamma_op(image)

    # The AdjustGamma operator: gamma parameter set to 20, gain parameter set to 16777216.
    with Image.open(dir_data()[4]) as image:
        adjust_gamma_op = vision.AdjustGamma(gamma=20.0, gain=-16777216)
        _ = adjust_gamma_op(image)

    # The AdjustGamma operator: gamma parameter set to 2.68, gain parameter set to 0.25.
    image = np.random.randint(0, 255, (30, 30, 3)).astype(np.uint8)
    to_pil_op = vision.ToPIL()
    image = to_pil_op(image)
    adjust_gamma_op = vision.AdjustGamma(gamma=2.68, gain=0.25)
    _ = adjust_gamma_op(image)

    # AdjustGamma Operator: Input data is in BMP format
    with   Image.open(dir_data()[3]) as image:
        adjust_gamma_op = vision.AdjustGamma(gamma=0.5, gain=-3.6)
        _ = adjust_gamma_op(image)


def test_adjust_gamma_exception_01():
    """
    Feature: AdjustGamma operation
    Description: Testing the AdjustGamma Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # AdjustGamma Operator: Test input data is 1D numpy data
    image = np.random.randint(0, 255, (30,)).astype(np.uint8)
    adjust_gamma_op = vision.AdjustGamma(gamma=1.2, gain=1.0)
    with pytest.raises(RuntimeError, match="AdjustGamma: input tensor is not in shape of <...,H,W,C> or <H,W>."):
        adjust_gamma_op(image)

    # AdjustGamma Operator: Test input data is a 4-channel numpy array.
    image = np.random.randint(0, 255, (10, 10, 4)).astype(np.uint8)
    adjust_gamma_op = vision.AdjustGamma(gamma=1.2, gain=1.0)
    with pytest.raises(RuntimeError, match="AdjustGamma: channel of input image should be 1 or 3."):
        adjust_gamma_op(image)

    # AdjustGamma Operator: Test input data is a 5-channel numpy array.
    image = np.random.randint(0, 255, (10, 3, 30, 5)).astype(np.uint8)
    adjust_gamma_op = vision.AdjustGamma(gamma=1.2, gain=1.0)
    with pytest.raises(RuntimeError, match="AdjustGamma: channel of input image should be 1 or 3."):
        adjust_gamma_op(image)

    # AdjustGamma Operator: Test input data is of type list
    image = list(np.random.randint(0, 255, (30, 30, 3)).astype(np.uint8))
    adjust_gamma_op = vision.AdjustGamma(gamma=1.2, gain=1.0)
    with pytest.raises(TypeError, match=r"Input should be NumPy or PIL image, got <class 'list'>."):
        adjust_gamma_op(image)

    # AdjustGamma Operator: Test input data is of type float
    image = 1.0
    adjust_gamma_op = vision.AdjustGamma(gamma=1.2, gain=1.0)
    with pytest.raises(TypeError, match=r"Input should be NumPy or PIL image, got <class 'float'>."):
        adjust_gamma_op(image)

    # AdjustGamma Operator: Test input data is 1D numpy data
    image = np.array(2.0)
    adjust_gamma_op = vision.AdjustGamma(gamma=1.2, gain=1.0)
    with pytest.raises(RuntimeError, match="AdjustGamma: input tensor is not in shape of <...,H,W,C> or <H,W>."):
        adjust_gamma_op(image)

    # AdjustGamma operator: No parameters passed
    with pytest.raises(TypeError, match="missing a required argument: 'gamma'"):
        vision.AdjustGamma()

    # AdjustGamma Operator: Passing Multiple Parameters
    with pytest.raises(TypeError, match="too many positional arguments"):
        vision.AdjustGamma(1.0, 1.0, 1.0)

    # # AdjustGamma operator: gamma parameter set to -0.05
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[0, 16777216\\]."):
        vision.AdjustGamma(gamma=-0.05)

    # AdjustGamma operator: gamma parameter set to 16777216.1
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[0, 16777216\\]."):
        vision.AdjustGamma(gamma=16777216.1)

    # AdjustGamma operator: gamma parameter set to True
    with pytest.raises(TypeError, match="Argument gamma with value True is not of type \\(<class"
                                        " 'float'>, <class 'int'>\\), but got <class 'bool'>."):
        vision.AdjustGamma(gamma=True)

    # AdjustGamma operator: gamma parameter set to [1]
    with pytest.raises(TypeError, match="Argument gamma with value \\[1\\] is not of type \\[<class "
                                        "'float'>, <class 'int'>\\], but got <class 'list'>."):
        vision.AdjustGamma(gamma=[1])

    # AdjustGamma operator: gamma parameter set to "1.0"
    with pytest.raises(TypeError, match="Argument gamma with value 1.0 is not of type \\[<class "
                                        "'float'>, <class 'int'>\\], but got <class 'str'>."):
        vision.AdjustGamma(gamma="1.0")

    # AdjustGamma operator: gain parameter set to 16777216.1
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[-16777216, 16777216\\]."):
        vision.AdjustGamma(gamma=1.2, gain=16777216.1)

    # AdjustGamma operator: gain parameter set to -16777216.1
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[-16777216, 16777216\\]."):
        vision.AdjustGamma(gamma=1.2, gain=-16777216.1)

    # AdjustGamma operator: gain parameter set to str
    with pytest.raises(TypeError, match="Argument gain with value 0.2 is not of type \\[<class"
                                        " 'float'>, <class 'int'>\\], but got <class 'str'>."):
        vision.AdjustGamma(gamma=1.2, gain="0.2")

    # AdjustGamma operator: gain parameter set to True
    with pytest.raises(TypeError, match="Argument gain with value True is not of type \\(<class"
                                        " 'float'>, <class 'int'>\\), but got <class 'bool'>."):
        vision.AdjustGamma(gamma=1.2, gain=True)

    # AdjustGamma operator: gain parameter set to tuple
    with pytest.raises(TypeError, match="Argument gain with value \\(0.5,\\) is not of type \\[<class"
                                        " 'float'>, <class 'int'>\\], but got <class 'tuple'>."):
        vision.AdjustGamma(gamma=1.2, gain=(0.5,))


if __name__ == "__main__":
    test_adjust_gamma_c_eager()
    test_adjust_gamma_py_eager()
    test_adjust_gamma_c_eager_gray()
    test_adjust_gamma_py_eager_gray()
    test_adjust_gamma_invalid_gamma_param_c()
    test_adjust_gamma_invalid_gamma_param_py()
    test_adjust_gamma_invalid_gain_param_c()
    test_adjust_gamma_invalid_gain_param_py()
    test_adjust_gamma_pipeline_c()
    test_adjust_gamma_pipeline_py()
    test_adjust_gamma_pipeline_py_gray()
    test_adjust_gamma_eager_image_type()
    test_adjust_gamma_eager_invalid_image_types1()
    test_adjust_gamma_eager_invalid_image_types2()
    test_adjust_gamma_operation_01()
    test_adjust_gamma_exception_01()
