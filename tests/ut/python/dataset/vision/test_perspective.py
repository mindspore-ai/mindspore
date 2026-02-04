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
Testing Perspective op in DE
"""
import cv2
import numpy as np
from numpy import random
from numpy.testing import assert_allclose
import os
import pytest
import PIL
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.transforms.transforms
import mindspore.dataset.transforms as t_trans
import mindspore.dataset.vision.transforms as vision
from mindspore.dataset.vision.utils import Inter
from mindspore import log as logger

DATA_DIR = "../data/dataset/testImageNetData/train/"

DATA_DIR_2 = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
TEST_DATA_DATASET_FUNC ="../data/dataset/"

image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")


def generate_numpy_random_rgb(shape):
    """
    Only generate floating points that are fractions like n / 256, since they
    are RGB pixels. Some low-precision floating point types in this test can't
    handle arbitrary precision floating points well.
    """
    return np.random.randint(0, 256, shape) / 255.


def test_perspective_python_implement():
    """
    Feature: Perspective
    Description: Test eager support for Perspective Python implementation
    Expectation: Return output image successfully
    """
    img_in = np.array([[[211, 192, 16], [146, 176, 190], [103, 86, 18], [23, 194, 246]],
                       [[17, 86, 38], [180, 162, 43], [197, 198, 224], [109, 3, 195]],
                       [[172, 197, 74], [33, 52, 136], [120, 185, 76], [105, 23, 221]],
                       [[197, 50, 36], [82, 187, 119], [124, 193, 164], [181, 8, 11]]], dtype=np.uint8)
    img_in = PIL.Image.fromarray(img_in)
    src = [[0, 63], [63, 63], [63, 0], [0, 0]]
    dst = [[0, 32], [32, 32], [32, 0], [0, 0]]
    perspective_op = vision.Perspective(src, dst, Inter.BILINEAR)
    img_ms = np.array(perspective_op(img_in))
    expect_result = np.array([[[139, 154, 71], [110, 122, 164], [0, 0, 0], [0, 0, 0]],
                              [[121, 122, 91], [129, 110, 120], [0, 0, 0], [0, 0, 0]],
                              [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                              [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=np.uint8)
    assert_allclose(img_ms.flatten(),
                    expect_result.flatten(),
                    rtol=1e-3,
                    atol=0)


def test_perspective_eager():
    """
    Feature: Perspective
    Description: Test eager support for Perspective Cpp implementation
    Expectation: Receive correct output image from op
    """

    # Eager 3-channel
    rgb_flat = generate_numpy_random_rgb((40000, 3)).astype(np.float32)
    img_in1 = rgb_flat.reshape((200, 200, 3))
    img_in1_cv = PIL.Image.fromarray(np.uint8(img_in1))
    img_width, img_height = 200, 200
    top_left = [random.randint(0, img_width - 1),
                random.randint(0, img_height - 1)]
    top_right = [random.randint(0, img_width - 1),
                 random.randint(0, img_height - 1)]
    bottom_right = [random.randint(0, img_width - 1),
                    random.randint(0, img_height - 1)]
    bottom_left = [random.randint(0, img_width-1),
                   random.randint(0, img_height - 1)]
    src = [[0, 0], [img_width - 1, 0], [img_width - 1, img_height - 1], [0, img_height - 1]]
    dst = [top_left, top_right, bottom_right, bottom_left]
    src_points = np.array(src, dtype="float32")
    dst_points = np.array(dst, dtype="float32")
    y = cv2.getPerspectiveTransform(src_points, dst_points)
    img_cv1 = cv2.warpPerspective(np.array(img_in1), y, img_in1_cv.size, cv2.INTER_LINEAR)
    perspective_op = vision.Perspective(src, dst, Inter.BILINEAR)
    img_ms1 = perspective_op(img_in1)
    assert_allclose(img_ms1.flatten(),
                    img_cv1.flatten(),
                    rtol=1e-3,
                    atol=0)


def test_perspective_invalid_param():
    """
    Feature: Perspective
    Description: Test Perspective implementation with invalid ignore parameter
    Expectation: Correct error is raised as expected
    """
    logger.info("Test Perspective implementation with invalid ignore parameter")
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(
            operations=[vision.Decode(), vision.Resize((224, 224)), lambda img: np.array(img[:, :, 0])],
            input_columns=["image"])
        # invalid list
        src = 1.0
        dst = 2.0
        data_set = data_set.map(operations=vision.Perspective(src, dst, Inter.BILINEAR),
                                input_columns="image")
    except TypeError as error:
        logger.info("Got an exception in Perspective: {}".format(str(error)))
        assert "is not of type [<class 'list'>, <class 'tuple'>], but got" in str(error)

    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(
            operations=[vision.Decode(), vision.Resize((224, 224)), lambda img: np.array(img[:, :, 0])],
            input_columns=["image"])
        # invalid list
        src = [2, 1, 3]
        dst = [1, 2, 3]
        data_set = data_set.map(operations=vision.Perspective(src, dst, Inter.BILINEAR),
                                input_columns="image")
    except TypeError as error:
        logger.info("Got an exception in Perspective: {}".format(str(error)))
        assert "Argument start_points[0] with value 2 is not of type [<class 'list'>, <class 'tuple'>]," in str(error)

    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(
            operations=[vision.Decode(), vision.Resize((224, 224)), lambda img: np.array(img[:, :, 0])],
            input_columns=["image"])
        # invalid list
        src = [[2], [1], [3]]
        dst = [[2], [1], [3]]
        data_set = data_set.map(operations=vision.Perspective(src, dst, Inter.BILINEAR),
                                input_columns="image")
    except TypeError as error:
        logger.info("Got an exception in Perspective: {}".format(str(error)))
        assert "start_points should be a list or tuple of length 4" in str(error)

    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(
            operations=[vision.Decode(), vision.Resize((224, 224)), lambda img: np.array(img[:, :, 0])],
            input_columns=["image"])
        # invalid list
        src = [[2], [1], [3], [4]]
        dst = [[2], [1], [3], [4]]
        data_set = data_set.map(operations=vision.Perspective(src, dst, Inter.BILINEAR),
                                input_columns="image")
    except TypeError as error:
        logger.info("Got an exception in Perspective: {}".format(str(error)))
        assert "start_points[0] should be a list or tuple of length 2" in str(error)

    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(
            operations=[vision.Decode(), vision.Resize((224, 224)), lambda img: np.array(img[:, :, 0])],
            input_columns=["image"])
        # invalid list
        src = [[2, 2], [1, 1], [3, 3], [4, 4]]
        dst = [[2, 2], [1, 1], [3, 3], [4, 2147483648]]
        data_set = data_set.map(operations=vision.Perspective(src, dst, Inter.BILINEAR),
                                input_columns="image")
    except ValueError as error:
        logger.info("Got an exception in Perspective: {}".format(str(error)))
        assert "Input end_points[3][1] is not within the required interval of [-2147483648, 2147483647]" in str(error)

    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(
            operations=[vision.Decode(), vision.Resize((224, 224)), lambda img: np.array(img[:, :, 0])],
            input_columns=["image"])
        # invalid list
        src = [[2, 2], [1, 1], [3, 3], [4, 2147483648]]
        dst = [[2, 2], [1, 1], [3, 3], [4, 4]]
        data_set = data_set.map(operations=vision.Perspective(src, dst, Inter.BILINEAR),
                                input_columns="image")
    except ValueError as error:
        logger.info("Got an exception in Perspective: {}".format(str(error)))
        assert "Input start_points[3][1] is not within the required interval of [-2147483648, 2147483647]" in str(error)


def test_perspective_invalid_interpolation():
    """
    Feature: Perspective
    Description: test Perspective with invalid interpolation
    Expectation: throw TypeError
    """
    logger.info("test_perspective_invalid_interpolation")
    dataset = ds.ImageFolderDataset(DATA_DIR, 1, shuffle=False, decode=True)
    try:
        src = [[0, 63], [63, 63], [63, 0], [0, 0]]
        dst = [[0, 63], [63, 63], [63, 0], [0, 0]]
        perspective_op = vision.Perspective(src, dst, interpolation="invalid")
        dataset.map(operations=perspective_op, input_columns=['image'])
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Argument interpolation with value invalid is not of type [<enum 'Inter'>]" in str(e)


def test_perspective_pipeline():
    """
    Feature: Perspective
    Description: Test Perspective C implementation Pipeline
    Expectation: Runs successfully
    """

    src = [[0, 63], [63, 63], [63, 0], [0, 0]]
    dst = [[0, 63], [63, 63], [63, 0], [0, 0]]

    # First dataset
    transforms1 = [vision.Decode(), vision.Resize([64, 64])]
    transforms1 = mindspore.dataset.transforms.transforms.Compose(
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
        vision.Perspective(src, dst, Inter.BILINEAR)
    ]
    transform2 = mindspore.dataset.transforms.transforms.Compose(
        transforms2)
    ds2 = ds.TFRecordDataset(DATA_DIR_2,
                             SCHEMA_DIR,
                             columns_list=["image"],
                             shuffle=False)
    ds2 = ds2.map(operations=transform2, input_columns=["image"])

    for data1, data2 in zip(ds1.create_dict_iterator(num_epochs=1),
                            ds2.create_dict_iterator(num_epochs=1)):
        ori_img = data1["image"].asnumpy()
        cvt_img = data2["image"].asnumpy()
        assert_allclose(ori_img.flatten(),
                        cvt_img.flatten(),
                        rtol=1e-5,
                        atol=0)
        assert ori_img.shape == cvt_img.shape


def test_perspective_operation_01():
    """
    Feature: Perspective operation
    Description: Testing the normal functionality of the Perspective operator
    Expectation: The Output is equal to the expected output
    """
    # Pipeline mode, parameter-free interpolation, Perspective interface invocation successful
    start_points = [[0, 63], [63, 63], [63, 0], [0, 0]]
    end_points = [[0, 32], [32, 32], [32, 0], [0, 0]]
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, 1)
    transforms = [
        vision.Decode(to_pil=True),
        vision.Perspective(start_points, end_points),
        vision.ToTensor()
    ]
    transform = t_trans.Compose(transforms)
    dataset = dataset.map(input_columns=["image"], operations=transform)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        break

    # In eager mode, the Perspective interface call succeeds when fed 3D uint8 data.
    image = np.random.randint(0, 128, (128, 128, 4)).astype(np.uint8)
    start_points = [[1, 1], [2, 2], [3, 3], [4, 4]]
    end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    perspective_op = vision.Perspective(start_points, end_points, Inter.PILCUBIC)
    _ = perspective_op(image)

    # In eager mode, the Perspective interface call succeeds when fed 2D uint16 data.
    image = np.random.randint(0, 128, (128, 128)).astype(np.uint16)
    start_points = [[10, 1], [2, 20], [30, 3], [40, 4]]
    end_points = [[20, 2], [10, 1], [3, 30], [4, 40]]
    perspective_op = vision.Perspective(start_points, end_points, Inter.NEAREST)
    perspective_op(image)

    # When input is 3D float32 data, the Perspective interface call succeeds.
    image = np.random.randn(5, 5, 3).astype(np.float32)
    start_points = [[1089, 1], [2, 202], [30, 3], [40, 402]]
    end_points = [[20, 2], [10, 1], [3, 303], [4, 4018]]
    perspective_op = vision.Perspective(start_points, end_points, Inter.PILCUBIC)
    perspective_op(image)

    # When input is 3D float64 data, the Perspective interface call succeeds.
    image = np.random.randn(255, 255, 255).astype(np.float64)
    start_points = ((1, 1), (2, 2), (3, 3), (4, 4))
    end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    perspective_op = vision.Perspective(start_points, end_points, Inter.AREA)
    perspective_op(image)

    # When the end_points parameter contains the maximum value, the Perspective interface call succeeds.
    image = np.random.randn(255, 255, 3)
    start_points = [[-100, 100], [200, -200], [300, 300], [-400, 400]]
    end_points = [[-2147483648, -2147483648], [-21474836, -21474836], [2147483647, 2147483647],
                  [2147483647, 2147483647]]
    perspective_op = vision.Perspective(start_points, end_points, Inter.CUBIC)
    perspective_op(image)

    # When the interpolation parameter is set to Inter.BICUBIC or Inter.CUBIC,
    # the Perspective interface call succeeds with identical precision.
    with Image.open(image_bmp) as image:
        start_points = [[-2147483648, -2147483648], [-21474836, -21474836], [2147483647, 2147483647],
                        [2147483647, 2147483647]]
        end_points = [[-2147483648, -2147483648], [-21474836, -21474836], [2147483647, 2147483647],
                      [2147483647, 2147483647]]
        perspective_op = vision.Perspective(start_points, end_points, Inter.BICUBIC)
        perspective_op1 = vision.Perspective(start_points, end_points, Inter.CUBIC)
        out = perspective_op(image)
        out1 = perspective_op1(image)
        assert out == out1

    # When the interpolation parameter is set to Inter.BILINEAR or Inter.LINEAR, the Perspective interface call succeeds
    with Image.open(image_gif) as image:
        start_points = [(-218, -21474), (-2147, -21474), (21474, 214748), (218, 21474)]
        end_points = [(-218, -214), (-2147, -2147), (2178, 21783), (218, 214)]
        perspective_op = vision.Perspective(start_points, end_points, Inter.BILINEAR)
        perspective_op1 = vision.Perspective(start_points, end_points, Inter.LINEAR)
        out = perspective_op(image)
        out1 = perspective_op1(image)
        assert out == out1

    # Pipeline mode: When inputting NumPy data, the Perspective interface call succeeds.
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    ds1 = ds.ImageFolderDataset(data_dir, 1)
    start_points = [[-28, -2474], [-214, -2144], [2147, 21448], [28, 2144]]
    end_points = [[-21, -24], [-217, -214], [218, 213], [21, 24]]
    transforms1 = [
        vision.Decode(to_pil=False),
        vision.Perspective(start_points, end_points),
    ]
    transform1 = t_trans.Compose(transforms1)
    ds1 = ds1.map(input_columns=["image"], operations=transform1)
    for _ in ds1.create_dict_iterator(output_numpy=True):
        pass


def test_perspective_exception_01():
    """
    Feature: Perspective operation
    Description: Testing the Perspective Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When the input image format is PNG, the Perspective interface call fails.
    with Image.open(image_png) as image:
        start_points = [[-28, -247], [-214, -214], [214, 218], (28, 214)]
        end_points = [(-21, -24), [-217, -214], [218, 213], [21, 24]]
        perspective_op = vision.Perspective(start_points, end_points, Inter.NEAREST)
        _ = perspective_op(image)

    # When input is PIL and the interpolation parameter is set to Inter.AREA, the Perspective interface call fails.
    with pytest.raises(TypeError, match="Current Interpolation is not supported with PIL input"):
        with Image.open(image_png) as image:
            start_points = [[-28, -2474], [-214, -2144], [2147, 21448], [28, 2144]]
            end_points = [[-21, -24], [-217, -214], [218, 213], [21, 24]]
            perspective_op = vision.Perspective(start_points, end_points, Inter.AREA)
            perspective_op(image)

    # When the input is PIL and the interpolation parameter is set to Inter.AREA, the Perspective interface call fails.
    with pytest.raises(TypeError, match="Current Interpolation is not supported with PIL input"):
        with Image.open(image_bmp) as image:
            start_points = [[-28, -2474], [-214, -2144], [2147, 21448], [28, 2144]]
            end_points = [[-21, -24], [-217, -214], [218, 213], [21, 24]]
            perspective_op = vision.Perspective(start_points, end_points, Inter.PILCUBIC)
            perspective_op(image)

    # When input is one-dimensional data, the Perspective interface call fails.
    with pytest.raises(RuntimeError, match=".*Perspective: the dimension of image tensor does not match the"
                                           " requirement of operator. Expecting tensor in dimension of \\(2, 3\\),"
                                           " in shape of <H, W> or <H, W, C>. But got dimension 1."
                                           " You may need to perform Decode first..*"):
        image = np.random.randn(255, ).astype(np.float64)
        start_points = ((1, 1), (2, 2), (3, 3), (4, 4))
        end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
        perspective_op = vision.Perspective(start_points, end_points, Inter.AREA)
        perspective_op(image)

    # When input is four-dimensional data, the Perspective interface call fails.
    with pytest.raises(RuntimeError, match=".*Perspective: the dimension of image tensor does not match the"
                                           " requirement of operator. Expecting tensor in dimension of \\(2, 3\\),"
                                           " in shape of <H, W> or <H, W, C>. But got dimension 4..*"):
        image = np.random.randn(128, 128, 128, 4)
        start_points = ((1, 1), (2, 2), (3, 3), (4, 4))
        end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
        perspective_op = vision.Perspective(start_points, end_points)
        perspective_op(image)

    # When input is tuple data, the Perspective interface call fails.
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'tuple'>."):
        image = tuple(np.random.randn(128, 128, 3))
        start_points = ((1, 1), (2, 2), (3, 3), (4, 4))
        end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
        perspective_op = vision.Perspective(start_points, end_points)
        perspective_op(image)

    # When input is list data, the Perspective interface call fails.
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        image = np.random.randint(0, 255, (128, 256, 3)).astype(np.uint8).tolist()
        start_points = ((1, 1), (2, 2), (3, 3), (4, 4))
        end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
        perspective_op = vision.Perspective(start_points, end_points)
        perspective_op(image)

    # When the start_points parameter is omitted, the Perspective interface call fails.
    image = np.random.randn(3, 4, 3)
    end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    interpolation = Inter.BILINEAR
    with pytest.raises(TypeError, match="missing a required argument: 'start_points'"):
        perspective_op = vision.Perspective(end_points=end_points, interpolation=interpolation)
        perspective_op(image)

    # When the end_points parameter is omitted, the Perspective interface call fails.
    image = np.random.randn(3, 4, 3)
    start_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    with pytest.raises(TypeError, match="missing a required argument: 'end_points'"):
        perspective_op = vision.Perspective(start_points=start_points)
        perspective_op(image)

    # When the list length of the start_points parameter is 3, the Perspective interface call fails.
    image = np.random.randn(3, 4, 3)
    start_points = [[2, 2], [1, 1], [3, 3]]
    end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    with pytest.raises(TypeError, match="start_points should be a list or tuple of length 4."):
        perspective_op = vision.Perspective(start_points=start_points, end_points=end_points)
        perspective_op(image)

    # When the start_points parameter is a list, the Perspective interface call fails.
    image = np.random.randn(3, 4, 3)
    start_points = [2, 2]
    end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    with pytest.raises(TypeError, match="Argument start_points\\[0\\] with value 2 is not of type \\[<class 'list'>,"
                                        " <class 'tuple'>\\], but got <class 'int'>."):
        perspective_op = vision.Perspective(start_points=start_points, end_points=end_points)
        perspective_op(image)


def test_perspective_exception_02():
    """
    Feature: Perspective operation
    Description: Testing the Perspective Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When the start_points parameter is a 1-tuple, the Perspective interface call fails.
    image = np.random.randn(3, 4, 3)
    start_points = ([2, 2])
    end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    with pytest.raises(TypeError, match="Argument start_points\\[0\\] with value 2 is not of type \\[<class 'list'>,"
                                        " <class 'tuple'>\\], but got <class 'int'>."):
        perspective_op = vision.Perspective(start_points=start_points, end_points=end_points)
        perspective_op(image)

    # When the start_points parameter is a str, the Perspective interface call fails.
    image = np.random.randn(3, 4, 3)
    start_points = [[2, 2], ['1', 1], [3, 3], [4, 4]]
    end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    with pytest.raises(TypeError, match="Argument start_points\\[1\\]\\[0\\] with value 1 is not of type "
                                        "\\[<class 'int'>\\], but got <class 'str'>."):
        perspective_op = vision.Perspective(start_points=start_points, end_points=end_points)
        perspective_op(image)

    # When the start_points parameter is a float, the Perspective interface call fails.
    image = np.random.randn(3, 4, 3)
    start_points = [[2, 2], [1, 1], [3, 3.0], [4, 4]]
    end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    with pytest.raises(TypeError, match="Argument start_points\\[2\\]\\[1\\] with value 3.0 is not of type "
                                        "\\[<class 'int'>\\], but got <class 'float'>."):
        perspective_op = vision.Perspective(start_points=start_points, end_points=end_points)
        perspective_op(image)

    # When the start_points parameter is a bool, the Perspective interface call fails.
    image = np.random.randn(3, 4, 3)
    start_points = True
    end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    with pytest.raises(TypeError, match="Argument start_points with value True is not of type \\[<class 'list'>,"
                                        " <class 'tuple'>\\], but got <class 'bool'>."):
        perspective_op = vision.Perspective(start_points=start_points, end_points=end_points)
        perspective_op(image)

    # The length of the sublist in the start_points parameter is not equal to 2
    image = np.random.randn(3, 4, 3)
    start_points = ([2, 0], [1, 1], [3], (4, 4))
    end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    with pytest.raises(TypeError, match="start_points\\[2\\] should be a list or tuple of length 2."):
        perspective_op = vision.Perspective(start_points=start_points, end_points=end_points)
        perspective_op(image)

    # When the list length of the end_points parameter is 3, the Perspective interface call fails.
    image = np.random.randn(3, 4, 3)
    start_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    end_points = [[2, 2], [1, 1], [3, 3]]
    with pytest.raises(TypeError, match="end_points should be a list or tuple of length 4."):
        perspective_op = vision.Perspective(start_points=start_points, end_points=end_points)
        perspective_op(image)

    # When the parameter end_points is a list, the Perspective interface call fails.
    image = np.random.randn(3, 4, 3)
    start_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    end_points = [2, 2]
    with pytest.raises(TypeError, match="Argument end_points\\[0\\] with value 2 is not of type \\[<class 'list'>,"
                                        " <class 'tuple'>\\], but got <class 'int'>."):
        perspective_op = vision.Perspective(start_points=start_points, end_points=end_points)
        perspective_op(image)

    # When the parameter end_points is a 1-tuple, the Perspective interface call fails.
    image = np.random.randn(3, 4, 3)
    start_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    end_points = ([2, 2])
    with pytest.raises(TypeError, match="Argument end_points\\[0\\] with value 2 is not of type \\[<class 'list'>,"
                                        " <class 'tuple'>\\], but got <class 'int'>."):
        perspective_op = vision.Perspective(start_points=start_points, end_points=end_points)
        perspective_op(image)

    # When the parameter end_points is of type str, the Perspective interface call fails.
    image = np.random.randn(3, 4, 3)
    start_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    end_points = [[2, 2], ['1', 1], [3, 3], [4, 4]]
    with pytest.raises(TypeError, match="Argument end_points\\[1\\]\\[0\\] with value 1 is not of type "
                                        "\\[<class 'int'>\\], but got <class 'str'>."):
        perspective_op = vision.Perspective(start_points=start_points, end_points=end_points)
        perspective_op(image)

    # When the parameter end_points is set to a float value, the Perspective interface call fails.
    image = np.random.randn(3, 4, 3)
    start_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    end_points = [[2, 2], [1, 1], [3, 3.0], [4, 4]]
    with pytest.raises(TypeError, match="Argument end_points\\[2\\]\\[1\\] with value 3.0 is not of type "
                                        "\\[<class 'int'>\\], but got <class 'float'>."):
        perspective_op = vision.Perspective(start_points=start_points, end_points=end_points)
        perspective_op(image)

    # When the parameter end_points is set to bool, the Perspective interface call fails.
    image = np.random.randn(3, 4, 3)
    start_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    end_points = True
    with pytest.raises(TypeError, match="Argument end_points with value True is not of type \\[<class 'list'>,"
                                        " <class 'tuple'>\\], but got <class 'bool'>."):
        perspective_op = vision.Perspective(start_points=start_points, end_points=end_points)
        perspective_op(image)


def test_perspective_exception_03():
    """
    Feature: Perspective operation
    Description: Testing the Perspective Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When the length of the sublist in the end_points parameter is not equal to 2, the Perspective interface call fails
    image = np.random.randn(3, 4, 3)
    start_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    end_points = ([2, 0], [1, 1], [3], (4, 4))
    with pytest.raises(TypeError, match="end_points\\[2\\] should be a list or tuple of length 2."):
        perspective_op = vision.Perspective(start_points=start_points, end_points=end_points)
        perspective_op(image)

    # The value in the parameter start_points is less than -2147483649.
    image = np.random.randn(3, 4, 3)
    start_points = [[2, 2], [1, 1], [3, 3], [4, -2147483649]]
    end_points = ([2, 0], [1, 1], [3, -2], (4, 4))
    with pytest.raises(ValueError, match="Input start_points\\[3\\]\\[1\\] is not within the required interval of "
                                         "\\[-2147483648, 2147483647\\]."):
        perspective_op = vision.Perspective(start_points=start_points, end_points=end_points)
        perspective_op(image)

    # The value in the start_points parameter exceeds 2147483647.
    image = np.random.randn(3, 4, 3)
    start_points = [[2, 2], [-21474836, 1], [3, 3], [4, 2147483648]]
    end_points = ([2, 0], [1, 1], [3, -2], (4, 4))
    with pytest.raises(ValueError, match="Input start_points\\[3\\]\\[1\\] is not within the required interval of "
                                         "\\[-2147483648, 2147483647\\]."):
        perspective_op = vision.Perspective(start_points=start_points, end_points=end_points)
        perspective_op(image)

    # The value in the parameter end_points is less than -2147483649.
    image = np.random.randn(3, 4, 3)
    start_points = ([2, 0], [1, 1], [3, -2], (4, 4))
    end_points = [[2, 2], [1, 1], [3, 3], [4, -2147483649]]
    with pytest.raises(ValueError, match="Input end_points\\[3\\]\\[1\\] is not within the required interval of "
                                         "\\[-2147483648, 2147483647\\]."):
        perspective_op = vision.Perspective(start_points=start_points, end_points=end_points)
        perspective_op(image)

    # The value in the parameter end_points exceeds 2147483647.
    image = np.random.randn(3, 4, 3)
    start_points = ([2, 0], [1, 1], [3, -2], (4, 4))
    end_points = [[2, 2], [-21474836, 1], [3, 3], [4, 2147483648]]
    with pytest.raises(ValueError, match="Input end_points\\[3\\]\\[1\\] is not within the required interval of "
                                         "\\[-2147483648, 2147483647\\]."):
        perspective_op = vision.Perspective(start_points=start_points, end_points=end_points)
        perspective_op(image)

    # The parameter interpolationMode is a boolean.
    image = np.random.randn(3, 4, 3)
    start_points = ([2, 0], [1, 1], [3, -2], (4, 4))
    end_points = [[2, 2], [-21478, 1], [3, 3], [4, 214788]]
    interpolation = True
    with pytest.raises(TypeError, match="Argument interpolation with value True is not of type \\[<enum 'Inter'>\\],"
                                        " but got <class 'bool'>."):
        perspective_op = vision.Perspective(start_points=start_points, end_points=end_points,
                                            interpolation=interpolation)
        perspective_op(image)

    # 参数interpolationMode为int时，Perspective接口调用失败
    # The parameter interpolationMode is an int.
    image = np.random.randn(3, 4, 3)
    start_points = ([2, 0], [1, 1], [3, -2], (4, 4))
    end_points = [[2, 2], [-21478, 1], [3, 3], [4, 214788]]
    interpolation = 1
    with pytest.raises(TypeError, match="Argument interpolation with value 1 is not of type \\[<enum 'Inter'>\\],"
                                        " but got <class 'int'>."):
        perspective_op = vision.Perspective(start_points=start_points, end_points=end_points,
                                            interpolation=interpolation)
        perspective_op(image)

    # The parameter interpolationMode is a str.
    image = np.random.randn(3, 4, 3)
    start_points = ([2, 0], [1, 1], [3, -2], (4, 4))
    end_points = [[2, 2], [-21478, 1], [3, 3], [4, 214788]]
    interpolation = 'Inter.AREA'
    with pytest.raises(TypeError, match="Argument interpolation with value Inter.AREA is not of type "
                                        "\\[<enum 'Inter'>\\], but got <class 'str'>."):
        perspective_op = vision.Perspective(start_points=start_points, end_points=end_points,
                                            interpolation=interpolation)
        perspective_op(image)

    # The parameter interpolationMode is a list.
    image = np.random.randn(3, 4, 3)
    start_points = ([2, 0], [1, 1], [3, -2], (4, 4))
    end_points = [[2, 2], [-21478, 1], [3, 3], [4, 214788]]
    interpolation = [Inter.BICUBIC]
    with pytest.raises(TypeError, match="Argument interpolation with value \\[<Inter.BICUBIC: 3>\\] is not of type "
                                        "\\[<enum 'Inter'>\\], but got <class 'list'>."):
        perspective_op = vision.Perspective(start_points=start_points, end_points=end_points,
                                            interpolation=interpolation)
        perspective_op(image)


if __name__ == "__main__":
    test_perspective_eager()
    test_perspective_invalid_param()
    test_perspective_invalid_interpolation()
    test_perspective_pipeline()
    test_perspective_operation_01()
    test_perspective_exception_01()
    test_perspective_exception_02()
    test_perspective_exception_03()
