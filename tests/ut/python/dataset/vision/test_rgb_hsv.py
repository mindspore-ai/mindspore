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
Testing RgbToHsv and HsvToRgb op in DE
"""

import colorsys
import numpy as np
from numpy.testing import assert_allclose
import os
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.transforms.transforms as t_trans
import mindspore.dataset.vision.transforms as vision
import mindspore.dataset.vision.py_transforms_util as util

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
TEST_DATA_DATASET_FUNC ="../data/dataset/"


def chw_to_hwc(img):
    """
    Transpose the input image; shape (C, H, W) to shape (H, W, C).

    Args:
        img (numpy.ndarray): Image to be converted.

    Returns:
        img (numpy.ndarray), Converted image.
    """
    res = np.transpose(img, (1, 2, 0))
    return res


def generate_numpy_random_rgb(shape):
    # Only generate floating points that are fractions like n / 256, since they
    # are RGB pixels. Some low-precision floating point types in this test can't
    # handle arbitrary precision floating points well.
    return np.random.randint(0, 256, shape) / 255.


def test_rgb_hsv_hwc():
    """
    Feature: RgbToHsv and HsvToRgb ops
    Description: Test RgbToHsv and HsvToRgb utilities with an image in HWC format
    Expectation: Output is equal to the expected output
    """
    rgb_flat = generate_numpy_random_rgb((64, 3)).astype(np.float32)
    rgb_np = rgb_flat.reshape((8, 8, 3))
    hsv_base = np.array([
        colorsys.rgb_to_hsv(
            r.astype(np.float64), g.astype(np.float64), b.astype(np.float64))
        for r, g, b in rgb_flat
    ])
    hsv_base = hsv_base.reshape((8, 8, 3))
    hsv_de = util.rgb_to_hsvs(rgb_np, True)
    assert hsv_base.shape == hsv_de.shape
    assert_allclose(hsv_base.flatten(), hsv_de.flatten(), rtol=1e-5, atol=0)

    hsv_flat = hsv_base.reshape(64, 3)
    rgb_base = np.array([
        colorsys.hsv_to_rgb(
            h.astype(np.float64), s.astype(np.float64), v.astype(np.float64))
        for h, s, v in hsv_flat
    ])
    rgb_base = rgb_base.reshape((8, 8, 3))
    rgb_de = util.hsv_to_rgbs(hsv_base, True)
    assert rgb_base.shape == rgb_de.shape
    assert_allclose(rgb_base.flatten(), rgb_de.flatten(), rtol=1e-5, atol=0)


def test_rgb_hsv_batch_hwc():
    """
    Feature: RgbToHsv and HsvToRgb ops
    Description: Test RgbToHsv and HsvToRgb utilities with a batch of images in HWC format
    Expectation: Output is equal to the expected output
    """
    rgb_flat = generate_numpy_random_rgb((64, 3)).astype(np.float32)
    rgb_np = rgb_flat.reshape((4, 2, 8, 3))
    hsv_base = np.array([
        colorsys.rgb_to_hsv(
            r.astype(np.float64), g.astype(np.float64), b.astype(np.float64))
        for r, g, b in rgb_flat
    ])
    hsv_base = hsv_base.reshape((4, 2, 8, 3))
    hsv_de = util.rgb_to_hsvs(rgb_np, True)
    assert hsv_base.shape == hsv_de.shape
    assert_allclose(hsv_base.flatten(), hsv_de.flatten(), rtol=1e-5, atol=0)

    hsv_flat = hsv_base.reshape((64, 3))
    rgb_base = np.array([
        colorsys.hsv_to_rgb(
            h.astype(np.float64), s.astype(np.float64), v.astype(np.float64))
        for h, s, v in hsv_flat
    ])
    rgb_base = rgb_base.reshape((4, 2, 8, 3))
    rgb_de = util.hsv_to_rgbs(hsv_base, True)
    assert rgb_de.shape == rgb_base.shape
    assert_allclose(rgb_base.flatten(), rgb_de.flatten(), rtol=1e-5, atol=0)


def test_rgb_hsv_chw():
    """
    Feature: RgbToHsv and HsvToRgb ops
    Description: Test RgbToHsv and HsvToRgb utilities with an image in CHW format
    Expectation: Output is equal to the expected output
    """
    rgb_flat = generate_numpy_random_rgb((64, 3)).astype(np.float32)
    rgb_np = rgb_flat.reshape((3, 8, 8))
    hsv_base = np.array([
        np.vectorize(colorsys.rgb_to_hsv)(
            rgb_np[0, :, :].astype(np.float64), rgb_np[1, :, :].astype(np.float64), rgb_np[2, :, :].astype(np.float64))
    ])
    hsv_base = hsv_base.reshape((3, 8, 8))
    hsv_de = util.rgb_to_hsvs(rgb_np, False)
    assert hsv_base.shape == hsv_de.shape
    assert_allclose(hsv_base.flatten(), hsv_de.flatten(), rtol=1e-5, atol=0)

    rgb_base = np.array([
        np.vectorize(colorsys.hsv_to_rgb)(
            hsv_base[0, :, :].astype(np.float64), hsv_base[1, :, :].astype(np.float64),
            hsv_base[2, :, :].astype(np.float64))
    ])
    rgb_base = rgb_base.reshape((3, 8, 8))
    rgb_de = util.hsv_to_rgbs(hsv_base, False)
    assert rgb_de.shape == rgb_base.shape
    assert_allclose(rgb_base.flatten(), rgb_de.flatten(), rtol=1e-5, atol=0)


def test_rgb_hsv_batch_chw():
    """
    Feature: RgbToHsv and HsvToRgb ops
    Description: Test RgbToHsv and HsvToRgb utilities with a batch of images in HWC format
    Expectation: Output is equal to the expected output
    """
    rgb_flat = generate_numpy_random_rgb((64, 3)).astype(np.float32)
    rgb_imgs = rgb_flat.reshape((4, 3, 2, 8))
    hsv_base_imgs = np.array([
        np.vectorize(colorsys.rgb_to_hsv)(
            img[0, :, :].astype(np.float64), img[1, :, :].astype(np.float64), img[2, :, :].astype(np.float64))
        for img in rgb_imgs
    ])
    hsv_de = util.rgb_to_hsvs(rgb_imgs, False)
    assert hsv_base_imgs.shape == hsv_de.shape
    assert_allclose(hsv_base_imgs.flatten(), hsv_de.flatten(), rtol=1e-5, atol=0)

    rgb_base = np.array([
        np.vectorize(colorsys.hsv_to_rgb)(
            img[0, :, :].astype(np.float64), img[1, :, :].astype(np.float64), img[2, :, :].astype(np.float64))
        for img in hsv_base_imgs
    ])
    rgb_de = util.hsv_to_rgbs(hsv_base_imgs, False)
    assert rgb_base.shape == rgb_de.shape
    assert_allclose(rgb_base.flatten(), rgb_de.flatten(), rtol=1e-5, atol=0)


def test_rgb_hsv_pipeline():
    """
    Feature: RgbToHsv and HsvToRgb ops
    Description: Test RgbToHsv and HsvToRgb ops in data pipeline
    Expectation: Output is equal to the expected output
    """
    # First dataset
    transforms1 = [
        vision.Decode(True),
        vision.Resize([64, 64]),
        vision.ToTensor()
    ]
    transforms1 = t_trans.Compose(transforms1)
    ds1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    ds1 = ds1.map(operations=transforms1, input_columns=["image"])

    # Second dataset
    transforms2 = [
        vision.Decode(True),
        vision.Resize([64, 64]),
        vision.ToTensor(),
        vision.RgbToHsv(),
        vision.HsvToRgb()
    ]
    transform2 = t_trans.Compose(transforms2)
    ds2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    ds2 = ds2.map(operations=transform2, input_columns=["image"])

    num_iter = 0
    for data1, data2 in zip(ds1.create_dict_iterator(num_epochs=1), ds2.create_dict_iterator(num_epochs=1)):
        num_iter += 1
        ori_img = data1["image"].asnumpy()
        cvt_img = data2["image"].asnumpy()
        assert_allclose(ori_img.flatten(), cvt_img.flatten(), rtol=1e-5, atol=0)
        assert ori_img.shape == cvt_img.shape


def test_hsv_to_rgb_operation_01():
    """
    Feature: HsvToRgb operation
    Description: Testing the normal functionality of the HsvToRgb operator
    Expectation: The Output is equal to the expected output
    """
    # HsvToRgb Operator: Test normally
    transforms = [
        vision.Decode(True),
        vision.Resize([64, 64]),
        vision.ToTensor(),
        vision.RgbToHsv(),
        vision.HsvToRgb()
    ]
    transform = t_trans.Compose(transforms)
    dataset = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    dataset = dataset.map(input_columns=["image"], operations=transform)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # HsvToRgb Operator: Normal testing, is_hwc=False
    transforms = [
        vision.Decode(True),
        vision.Resize([64, 64]),
        vision.ToTensor(),
        vision.HsvToRgb(is_hwc=False)
    ]
    transform = t_trans.Compose(transforms)
    dataset = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    dataset = dataset.map(input_columns=["image"], operations=transform)

    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # HsvToRgb Operator: Normal testing, is_hwc=False
    transforms = [
        vision.Decode(True),
        vision.Resize([64, 64]),
        vision.ToTensor(),
        chw_to_hwc,
        vision.HsvToRgb(is_hwc=True),
        vision.HWC2CHW()
    ]
    transform = t_trans.Compose(transforms)
    dataset = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    dataset = dataset.map(input_columns=["image"], operations=transform)

    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # HsvToRgb Operator: Normal testing, eager mode, jpg image
    image_file = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    is_hwc = True
    hsv_to_rgb_op = vision.HsvToRgb(is_hwc)
    with Image.open(image_file) as image:
        _ = hsv_to_rgb_op(np.array(image))

    # HsvToRgb Operator: Normal testing, eager mode, bmp image
    image_file3 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
    is_hwc = True
    hsv_to_rgb_op = vision.HsvToRgb(is_hwc)
    with Image.open(image_file3) as image:
        _ = hsv_to_rgb_op(np.array(image))


def test_hsv_to_rgb_operation_02():
    """
    Feature: HsvToRgb operation
    Description: Testing the normal functionality of the HsvToRgb operator
    Expectation: The Output is equal to the expected output
    """
    # HsvToRgb Operator: Normal testing, eager mode, jpg image, is_hwc=False
    image_file = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    is_hwc = False
    hsv_to_rgb_op = vision.HsvToRgb(is_hwc)
    with Image.open(image_file) as image:
        _ = hsv_to_rgb_op(np.array(image).transpose(2, 0, 1))

    # HsvToRgb Operator: Normal testing, is_hwc default value
    hsv_to_rgb_op1 = vision.HsvToRgb()
    hsv_to_rgb_op2 = vision.HsvToRgb(is_hwc=False)
    image_file = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_file) as image:
        out1 = hsv_to_rgb_op1(np.array(image).transpose(2, 0, 1))
        out2 = hsv_to_rgb_op2(np.array(image).transpose(2, 0, 1))
        assert (out1 == out2).all()

    # HsvToRgb Operator: Normal testing, input data is a 3D numpy uint8 array <32, 32, 3>
    image = np.random.randint(0, 255, (32, 32, 3)).astype(np.uint8)
    is_hwc = True
    hsv_to_rgb_op = vision.HsvToRgb(is_hwc)
    _ = hsv_to_rgb_op(np.array(image))


def test_hsv_to_rgb_exception_01():
    """
    Feature: HsvToRgb operation
    Description: Testing the HsvToRgb Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # HsvToRgb Operator: Anomaly Testing, Eager Mode, 4-Channel PNG Image
    image_file2 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    is_hwc = True
    hsv_to_rgb_op = vision.HsvToRgb(is_hwc)
    with Image.open(image_file2) as image:
        with pytest.raises(TypeError, match=r'img should be 3 channels RGB img. Got 4 channels.'):
            hsv_to_rgb_op(np.array(image))

    # HsvToRgb Operator: Error Testing, Eager Mode, GIF Image <H,W>
    image_file1 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")
    is_hwc = False
    hsv_to_rgb_op = vision.HsvToRgb(is_hwc)
    with Image.open(image_file1) as image:
        with pytest.raises(TypeError,
                           match=r'img shape should be \(H, W, C\)\/\(N, H, W, C\)\/\(C, H, W\)\/\(N, C, H, W\).'
                                 r' Got \(283, 212\).'):
            hsv_to_rgb_op(np.array(image))

    # HsvToRgb Operator: Exception Test, is_hwc Type Error: float
    is_hwc = -0.5
    with pytest.raises(TypeError, match=r'is not of type \[\<class \'bool\'\>\]'):
        vision.HsvToRgb(is_hwc)

    # HsvToRgb Operator: Exception Test, is_hwc Type Error: tuple
    is_hwc = (0.2, 0.5)
    with pytest.raises(TypeError, match=r'is not of type \[\<class \'bool\'\>\]'):
        vision.HsvToRgb(is_hwc)

    # HsvToRgb Operator: Exception Test, is_hwc Type Error: list
    is_hwc = [0.2, 0.5]
    with pytest.raises(TypeError, match=r'is not of type \[\<class \'bool\'\>\]'):
        vision.HsvToRgb(is_hwc)

    # HsvToRgb Operator: Anomaly Testing, input data is a 3D numpy uint8 array <32, 32, 1>
    in_c = 1
    image = np.random.randint(0, 255, (32, 32, in_c)).astype(np.uint8)
    is_hwc = False
    hsv_to_rgb_op = vision.HsvToRgb(is_hwc)
    with pytest.raises(TypeError, match=r'img should be 3 channels RGB img'):
        hsv_to_rgb_op(np.array(image))

    # HsvToRgb Operator: Anomaly Testing, input data is a 3D numpy uint8 array <32, 32, 2>
    in_c = 2
    image = np.random.randint(0, 255, (32, 32, in_c)).astype(np.uint8)
    is_hwc = False
    hsv_to_rgb_op = vision.HsvToRgb(is_hwc)
    with pytest.raises(TypeError, match=r'img should be 3 channels RGB img'):
        hsv_to_rgb_op(np.array(image))

    # HsvToRgb Operator: Anomaly Testing, input data is a 3D numpy uint8 array <32, 32, 4>
    in_c = 4
    image = np.random.randint(0, 255, (32, 32, in_c)).astype(np.uint8)
    is_hwc = False
    hsv_to_rgb_op = vision.HsvToRgb(is_hwc)
    with pytest.raises(TypeError, match=r'img should be 3 channels RGB img'):
        hsv_to_rgb_op(np.array(image))


def test_rgb_to_hsv_operation_01():
    """
    Feature: RgbToHsv operation
    Description: Testing the normal functionality of the RgbToHsv operator
    Expectation: The Output is equal to the expected output
    """
    # Test RgbToHsv func to convert ten Numpy RGB images to HSV images.
    transforms = [
        vision.Decode(),
        vision.Resize([64, 64]),
        vision.ToTensor(),
        vision.RgbToHsv()
    ]
    transforms = t_trans.Compose(transforms)
    dataset = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    dataset = dataset.map(input_columns=["image"], operations=transforms)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RgbToHsv func ten images and is_hwc=False.
    transforms2 = [
        vision.Decode(),
        vision.Resize([64, 64]),
        vision.ToTensor(),
        vision.RgbToHsv(is_hwc=False)
    ]
    transform = t_trans.Compose(transforms2)
    dataset = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    dataset = dataset.map(input_columns=["image"], operations=transform)

    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RgbToHsv func ten images and is_hwc=True.
    transforms2 = [
        vision.Decode(),
        vision.Resize([64, 64]),
        vision.ToTensor(),
        chw_to_hwc,
        vision.RgbToHsv(is_hwc=True),
        vision.HWC2CHW()
    ]
    transform = t_trans.Compose(transforms2)
    dataset = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    dataset = dataset.map(input_columns=["image"], operations=transform)

    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RgbToHsv func normal: image_path
    image_file = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    image = Image.open(image_file)
    is_hwc = True
    rgb_to_hsv_op = vision.RgbToHsv(is_hwc)
    _ = rgb_to_hsv_op(np.array(image))
    image.close()


def test_rgb_to_hsv_operation_02():
    """
    Feature: RgbToHsv operation
    Description: Testing the normal functionality of the RgbToHsv operator
    Expectation: The Output is equal to the expected output
    """
    # Test RgbToHsv func normal: image_path
    image_file3 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
    image = Image.open(image_file3)
    is_hwc = True
    rgb_to_hsv_op = vision.RgbToHsv(is_hwc)
    _ = rgb_to_hsv_op(np.array(image))
    image.close()

    # Test RgbToHsv func normal: [0.5, 1.5]
    image_file = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    image = Image.open(image_file)
    is_hwc = False
    rgb_to_hsv_op = vision.RgbToHsv(is_hwc)
    _ = rgb_to_hsv_op(np.array(image).transpose(2, 0, 1))
    image.close()

    # Test RgbToHsv func normal: is_hwc default
    image_file = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    image = Image.open(image_file)
    rgb_to_hsv_op = vision.RgbToHsv()
    _ = rgb_to_hsv_op(np.array(image).transpose(2, 0, 1))
    image.close()

    # Test RgbToHsv func normal:  three channel
    image = np.random.randint(0, 255, (32, 32, 3)).astype(np.uint8)
    is_hwc = True
    rgb_to_hsv_op = vision.RgbToHsv(is_hwc)
    _ = rgb_to_hsv_op(np.array(image))

    # Test RgbToHsv func normal: test Numpy format
    image = np.random.randn(23, 43, 3)
    is_hwc = True
    rgb_to_hsv_op = vision.RgbToHsv(is_hwc)
    _ = rgb_to_hsv_op(np.array(image))

    # Test RgbToHsv func normal: test tolist
    image = np.random.randn(23, 43, 3).tolist()
    is_hwc = True
    rgb_to_hsv_op = vision.RgbToHsv(is_hwc)
    _ = rgb_to_hsv_op(np.array(image))

    # Test RgbToHsv func normal: test list format
    image = list(np.random.randn(23, 43, 3))
    is_hwc = True
    rgb_to_hsv_op = vision.RgbToHsv(is_hwc)
    _ = rgb_to_hsv_op(np.array(image))

    # Test RgbToHsv func normal: test Numpy format, 2-D
    image = np.random.randn(23, 43)
    is_hwc = True
    rgb_to_hsv_op = vision.RgbToHsv(is_hwc)
    with pytest.raises(TypeError, match="img shape should be \\(H, W, C\\)/\\(N, H, W, C\\)/"
                                        "\\(C ,H, W\\)/\\(N, C, H, W\\)."):
        rgb_to_hsv_op(np.array(image))

    # Test RgbToHsv func normal: test Numpy format, 4-D
    image = np.random.randn(23, 43, 3, 3)
    is_hwc = True
    rgb_to_hsv_op = vision.RgbToHsv(is_hwc)
    _ = rgb_to_hsv_op(np.array(image))


def test_rgb_to_hsv_exception_01():
    """
    Feature: RgbToHsv operation
    Description: Testing the RgbToHsv Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Test RgbToHsv func normal: image_path
    image_file2 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    image = Image.open(image_file2)
    is_hwc = True
    rgb_to_hsv_op = vision.RgbToHsv(is_hwc)
    with pytest.raises(TypeError, match=r'img should be 3 channels RGB img. Got 4 channels.'):
        rgb_to_hsv_op(np.array(image))
        image.close()

    # Test RgbToHsv func normal: image_file1
    image_file1 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")
    image = Image.open(image_file1)
    is_hwc = False
    rgb_to_hsv_op = vision.RgbToHsv(is_hwc)
    with pytest.raises(TypeError, match=r'img shape should be \(H, W, C\)/\(N, H, W, C\)/\(C ,H, W\)/\(N, C, H, W\).'):
        rgb_to_hsv_op(np.array(image))
        image.close()

    # Test RgbToHsv func normal: -0.5
    is_hwc = -0.5
    with pytest.raises(TypeError, match=r'is not of type \[\<class \'bool\'\>]'):
        vision.RgbToHsv(is_hwc)

    # Test RgbToHsv func normal: 0
    is_hwc = 0
    with pytest.raises(TypeError, match=r'is not of type \[\<class \'bool\'\>]'):
        vision.RgbToHsv(is_hwc)

    # Test RgbToHsv func normal: 1
    is_hwc = 1
    with pytest.raises(TypeError, match=r'is not of type \[\<class \'bool\'\>]'):
        vision.RgbToHsv(is_hwc)

    # Test RgbToHsv func normal: (0.2, 0.5)
    is_hwc = (0.2, 0.5)
    with pytest.raises(TypeError, match=r'is not of type \[\<class \'bool\'\>]'):
        vision.RgbToHsv(is_hwc)

    # Test RgbToHsv func normal: [0.2, 0.5]
    is_hwc = [0.2, 0.5]
    with pytest.raises(TypeError, match=r'is not of type \[\<class \'bool\'\>]'):
        vision.RgbToHsv(is_hwc)

    # Test RgbToHsv func normal:  3
    is_hwc = 3
    with pytest.raises(TypeError, match=r'is not of type \[\<class \'bool\'\>]'):
        vision.RgbToHsv(is_hwc)

    # Test RgbToHsv func normal:  3.0
    is_hwc = "3.0"
    with pytest.raises(TypeError, match=r'Argument is_hwc with value 3.0 is not of type \[\<class \'bool\'\>]'):
        vision.RgbToHsv(is_hwc)

    # Test RgbToHsv func normal: one channel   "in_c" is 1
    in_c = 1
    image = np.random.randint(0, 255, (32, 32, in_c)).astype(np.uint8)
    is_hwc = False
    rgb_to_hsv_op = vision.RgbToHsv(is_hwc)
    with pytest.raises(TypeError, match=r'img should be 3 channels RGB img'):
        rgb_to_hsv_op(np.array(image))

    # Test RgbToHsv func normal: two channel "in_c" is 2
    in_c = 2
    image = np.random.randint(0, 255, (32, 32, in_c)).astype(np.uint8)
    is_hwc = False
    rgb_to_hsv_op = vision.RgbToHsv(is_hwc)
    with pytest.raises(TypeError, match=r'img should be 3 channels RGB img'):
        rgb_to_hsv_op(np.array(image))

    # Test RgbToHsv func normal: four channel "in_c" is 4
    in_c = 4
    image = np.random.randint(0, 255, (32, 32, in_c)).astype(np.uint8)
    is_hwc = False
    rgb_to_hsv_op = vision.RgbToHsv(is_hwc)
    with pytest.raises(TypeError, match=r'img should be 3 channels RGB img'):
        rgb_to_hsv_op(np.array(image))


if __name__ == "__main__":
    test_rgb_hsv_hwc()
    test_rgb_hsv_batch_hwc()
    test_rgb_hsv_chw()
    test_rgb_hsv_batch_chw()
    test_rgb_hsv_pipeline()
    test_hsv_to_rgb_operation_01()
    test_hsv_to_rgb_operation_02()
    test_hsv_to_rgb_exception_01()
    test_rgb_to_hsv_operation_01()
    test_rgb_to_hsv_operation_02()
    test_rgb_to_hsv_exception_01()
