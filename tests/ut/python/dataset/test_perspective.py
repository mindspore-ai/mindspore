# Copyright 2022 Huawei Technologies Co., Ltd
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
import PIL

import mindspore.dataset as ds
import mindspore.dataset.transforms.transforms
import mindspore.dataset.vision.transforms as vision
from mindspore.dataset.vision.utils import Inter
from mindspore import log as logger

DATA_DIR = "../data/dataset/testImageNetData/train/"
MNIST_DATA_DIR = "../data/dataset/testMnistData"

DATA_DIR_2 = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

IMAGE_FILE = "../data/dataset/apple.jpg"


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


if __name__ == "__main__":
    test_perspective_eager()
    test_perspective_invalid_param()
    test_perspective_invalid_interpolation()
    test_perspective_pipeline()
