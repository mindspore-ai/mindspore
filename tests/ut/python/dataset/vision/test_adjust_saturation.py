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
Testing AdjustSaturation op in DE
"""
import numpy as np
from numpy.testing import assert_allclose
import os
import PIL
from PIL import Image, ImageEnhance
import pytest

import mindspore.dataset as ds
import mindspore.dataset.transforms.transforms as t_trans
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger


DATA_DIR = "../data/dataset/testImageNetData/train/"
DATA_DIR_2 = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
IMAGE_FILE = "../data/dataset/apple.jpg"

TEST_DATA_DATASET_FUNC ="../data/dataset/"


def dir_data():
    """ 获取数据集路径 """
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


def test_adjust_saturation_eager():
    """
    Feature: AdjustSaturation op
    Description: Test eager support for AdjustSaturation C implementation
    Expectation: Output is the same as expected output
    """
    # Eager 3-channel
    rgb_flat = generate_numpy_random_rgb((64, 3)).astype(np.uint8)
    img_in = rgb_flat.reshape((8, 8, 3))
    img_pil = Image.fromarray(img_in)

    adjustsaturation_op = vision.AdjustSaturation(0.0)
    img_out = adjustsaturation_op(img_in)
    pil_out = ImageEnhance.Color(img_pil).enhance(0)
    pil_out = np.array(pil_out)
    assert_allclose(pil_out.flatten(),
                    img_out.flatten(),
                    rtol=1e-5,
                    atol=0)

    img_in2 = PIL.Image.open("../data/dataset/apple.jpg").convert("RGB")

    adjustsaturation_op2 = vision.AdjustSaturation(1.0)
    img_out2 = adjustsaturation_op2(img_in2)
    img_out2 = np.array(img_out2)
    pil_out2 = ImageEnhance.Color(img_in2).enhance(1)
    pil_out2 = np.array(pil_out2)
    assert_allclose(pil_out2.flatten(),
                    img_out2.flatten(),
                    rtol=1e-5,
                    atol=0)


def test_adjust_saturation_invalid_saturationfactor_param():
    """
    Feature: AdjustSaturation op
    Description: Test AdjustSaturation Cpp implementation with invalid ignore parameter
    Expectation: Correct error is raised as expected
    """
    logger.info("Test AdjustSaturationC implementation with invalid ignore parameter")
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(
            operations=[vision.Decode(), vision.Resize((224, 224)), lambda img: np.array(img[:, :, 0])],
            input_columns=["image"])
        # invalid alpha
        data_set = data_set.map(operations=vision.AdjustSaturation(saturation_factor=-10.0),
                                input_columns="image")
    except ValueError as error:
        logger.info("Got an exception in AdjustSaturation: {}".format(str(error)))
        assert "Input is not within the required interval of " in str(error)
    try:
        data_set = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
        data_set = data_set.map(
            operations=[vision.Decode(), vision.Resize((224, 224)), lambda img: np.array(img[:, :, 0])],
            input_columns=["image"])
        # invalid alpha
        data_set = data_set.map(operations=vision.AdjustSaturation(saturation_factor=[1.0, 2.0]),
                                input_columns="image")
    except TypeError as error:
        logger.info("Got an exception in AdjustSaturation: {}".format(str(error)))
        assert "is not of type [<class 'float'>, <class 'int'>], but got" in str(error)


def test_adjust_saturation_pipeline():
    """
    Feature: AdjustSaturation op
    Description: Test AdjustSaturation Cpp implementation Pipeline
    Expectation: Output is the same as expected output
    """
    # First dataset
    transforms1 = [vision.Decode(), vision.Resize([64, 64])]
    transforms1 = t_trans.Compose(transforms1)
    ds1 = ds.TFRecordDataset(DATA_DIR_2,
                             SCHEMA_DIR,
                             columns_list=["image"],
                             shuffle=False)
    ds1 = ds1.map(operations=transforms1, input_columns=["image"])

    # Second dataset
    transforms2 = [
        vision.Decode(),
        vision.Resize([64, 64]),
        vision.AdjustSaturation(1.0)
    ]
    transform2 = t_trans.Compose(transforms2)
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


def test_adjust_saturation_operation_01():
    """
    Feature: AdjustSaturation operation
    Description: Testing the normal functionality of the AdjustSaturation operator
    Expectation: The Output is equal to the expected output
    """
    # AdjustSaturation Operator: Test Pipeline Mode
    transform1 = [vision.Decode()]
    transform1 = t_trans.Compose(transform1)
    ds1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    ds1 = ds1.map(operations=transform1, input_columns=["image"])
    transform2 = [vision.Decode(), vision.AdjustSaturation(saturation_factor=2.0)]
    transform2 = t_trans.Compose(transform2)
    ds2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    ds2 = ds2.map(operations=transform2, input_columns=["image"])
    for _ in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        pass

    # AdjustSaturation Operator: Input data is in JPG format
    with Image.open(dir_data()[1]) as image:
        adjust_saturation_op = vision.AdjustSaturation(saturation_factor=2.0)
        _ = adjust_saturation_op(image)

    # AdjustSaturation Operator: Input data is in GIF format
    with Image.open(dir_data()[2]) as image:
        adjust_saturation_op = vision.AdjustSaturation(saturation_factor=2.0)
        with pytest.raises(ValueError, match=r"image has wrong mode"):
            adjust_saturation_op(image)

    # AdjustSaturation Operator: Input data is in BMP format
    with  Image.open(dir_data()[3]) as image:
        adjust_saturation_op = vision.AdjustSaturation(saturation_factor=2.0)
        _ = adjust_saturation_op(image)

    # AdjustSaturation Operator: Input data is in PNG format
    with Image.open(dir_data()[4]) as image:
        adjust_saturation_op = vision.AdjustSaturation(saturation_factor=2.0)
        _ = adjust_saturation_op(image)

    # AdjustSaturation operator: Input data is a 3D numpy array.
    image = np.random.randint(0, 255, (30, 30, 3)).astype(np.uint8)
    adjust_saturation_op = vision.AdjustSaturation(saturation_factor=2.0)
    _ = adjust_saturation_op(image)

    # AdjustSaturation operator: The saturation_factor parameter is of type int.
    image = np.random.randint(0, 255, (30, 30, 3)).astype(np.uint8)
    adjust_saturation_op = vision.AdjustSaturation(saturation_factor=2)
    _ = adjust_saturation_op(image)


def test_adjust_saturation_exception_01():
    """
    Feature: AdjustSaturation operation
    Description: Testing the AdjustSaturation Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # AdjustSaturation operator: Input is 4-dimensional numpy data.
    image = np.random.randint(0, 255, (30, 30, 3, 3)).astype(np.uint8)
    adjust_saturation_op = vision.AdjustSaturation(saturation_factor=2.0)
    with pytest.raises(RuntimeError) as e:
        adjust_saturation_op(image)
    assert ("the dimension of image tensor does not match the requirement of operator. "
            "Expecting tensor in dimension of (3), in shape of <H, W, C>. But got dimension 4.") in str(e.value)

    # AdjustSaturation operator: Input is 2-dimensional numpy data.
    image = np.random.randint(0, 255, (30, 30)).astype(np.uint8)
    adjust_saturation_op = vision.AdjustSaturation(saturation_factor=2.0)
    with pytest.raises(RuntimeError) as e:
        adjust_saturation_op(image)
    assert ("the dimension of image tensor does not match the requirement of operator. "
            "Expecting tensor in dimension of (3), in shape of <H, W, C>. But got dimension 2") in str(e.value)

    # AdjustSaturation operator: Input is 1-dimensional numpy data
    image = np.random.randint(0, 255, (3)).astype(np.uint8)
    adjust_saturation_op = vision.AdjustSaturation(saturation_factor=2.0)
    with pytest.raises(RuntimeError) as e:
        adjust_saturation_op(image)
    assert ("the dimension of image tensor does not match the requirement of operator. Expecting tensor in dimension "
            "of (3), in shape of <H, W, C>. But got dimension 1. You may need to perform Decode first") in str(e.value)

    # AdjustSaturation Operator: Input data channel is 4
    image = np.random.randint(0, 255, (30, 30, 4)).astype(np.uint8)
    adjust_saturation_op = vision.AdjustSaturation(saturation_factor=2.0)
    with pytest.raises(RuntimeError) as e:
        adjust_saturation_op(image)
    assert ("the channel of image tensor does not match the requirement of operator. "
            "Expecting tensor in channel of (3). But got channel 4") in str(e.value)

    # AdjustSaturation Operator: Input data type is int
    image = 10
    adjust_saturation_op = vision.AdjustSaturation(saturation_factor=2.0)
    with pytest.raises(TypeError,
                       match=r"Input should be NumPy or PIL image, got <class 'int'>."):
        adjust_saturation_op(image)

    # AdjustSaturation Operator: Input data type is list
    image = (np.random.randint(0, 255, (30, 30, 3)).astype(np.uint8)).tolist()
    adjust_saturation_op = vision.AdjustSaturation(saturation_factor=2.0)
    with pytest.raises(TypeError,
                       match=r"Input should be NumPy or PIL image, got <class 'list'>."):
        adjust_saturation_op(image)

    # AdjustSaturation Operator: Input data type is list[numpy,numpy,...]
    image = list(np.random.randint(0, 255, (30, 30, 3)).astype(np.uint8))
    adjust_saturation_op = vision.AdjustSaturation(saturation_factor=2.0)
    with pytest.raises(TypeError,
                       match=r"Input should be NumPy or PIL image, got <class 'list'>."):
        adjust_saturation_op(image)

    # AdjustSaturation Operator: Input data type is tuple
    image = tuple(np.random.randint(0, 255, (30, 30, 3)).astype(np.uint8))
    adjust_saturation_op = vision.AdjustSaturation(saturation_factor=2.0)
    with pytest.raises(TypeError,
                       match=r"Input should be NumPy or PIL image, got <class 'tuple'>."):
        adjust_saturation_op(image)

    # AdjustSaturation Operator: Input data type is float
    image = 10.
    adjust_saturation_op = vision.AdjustSaturation(saturation_factor=2.0)
    with pytest.raises(TypeError,
                       match=r"Input should be NumPy or PIL image, got <class 'float'>."):
        adjust_saturation_op(image)

    # AdjustSaturation Operator: Input data type is uint16
    image = np.random.randint(0, 255, (30, 30, 3)).astype(np.uint16)
    adjust_saturation_op = vision.AdjustSaturation(saturation_factor=2.0)
    _ = adjust_saturation_op(image)

    # AdjustSaturation Operator: Input data type is str
    image = np.random.randint(0, 255, (30, 30, 3)).astype(np.str_)
    adjust_saturation_op = vision.AdjustSaturation(saturation_factor=2.0)
    with pytest.raises(RuntimeError,
                       match=r"Expecting tensor in type of \(uint8, uint16, float32\). But got type string"):
        adjust_saturation_op(image)

    # AdjustSaturation Operator: Input data type is uint32
    image = np.random.randint(0, 255, (30, 30, 3)).astype(np.uint32)
    adjust_saturation_op = vision.AdjustSaturation(saturation_factor=2.0)
    with pytest.raises(RuntimeError,
                       match=r"Expecting tensor in type of \(uint8, uint16, float32\). But got type uint32"):
        adjust_saturation_op(image)

    # AdjustSaturation Operator: Input data type is float64
    image = np.random.randint(0, 255, (30, 30, 3)).astype(np.float64)
    adjust_saturation_op = vision.AdjustSaturation(saturation_factor=2.0)
    with pytest.raises(RuntimeError,
                       match=r"Expecting tensor in type of \(uint8, uint16, float32\). But got type float64."):
        adjust_saturation_op(image)

    # AdjustSaturation Operator: Input data type is int64
    image = np.random.randint(0, 255, (30, 30, 3)).astype(np.int64)
    adjust_saturation_op = vision.AdjustSaturation(saturation_factor=2.0)
    with pytest.raises(RuntimeError,
                       match=r"Expecting tensor in type of \(uint8, uint16, float32\). But got type int64"):
        adjust_saturation_op(image)


def test_adjust_saturation_exception_02():
    """
    Feature: AdjustSaturation operation
    Description: Testing the AdjustSaturation Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # AdjustSaturation operator: The saturation_factor parameter is of type string
    with pytest.raises(TypeError,
                       match=r"Argument saturation_factor with value 1 is not of "
                             r"type \[<class 'float'>, <class 'int'>\], but got <class 'str'>."):
        vision.AdjustSaturation(saturation_factor='1')

    # AdjustSaturation operator: The saturation_factor parameter is True
    with pytest.raises(TypeError,
                       match=r"Argument saturation_factor with value True is not "
                             r"of type \(<class 'float'>, <class 'int'>\), but got <class 'bool'>"):
        vision.AdjustSaturation(saturation_factor=True)

    # AdjustSaturation operator: The saturation_factor parameter is None
    with pytest.raises(TypeError,
                       match=r"Argument saturation_factor with value None is not of "
                             r"type \[<class 'float'>, <class 'int'>\], but got <class 'NoneType'>."):
        vision.AdjustSaturation(saturation_factor=None)


if __name__ == "__main__":
    test_adjust_saturation_eager()
    test_adjust_saturation_invalid_saturationfactor_param()
    test_adjust_saturation_pipeline()
    test_adjust_saturation_operation_01()
    test_adjust_saturation_exception_01()
    test_adjust_saturation_exception_02()
