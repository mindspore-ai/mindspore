# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""Test Eager Support for Vision ops in Dataset"""
import os
import sys
import time
import copy
import cv2
import numpy as np
import pytest

from PIL import Image

from mindspore import log as logger
from mindspore.dataset.vision import Inter
import mindspore as ms
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms

# pylint: disable=W0212
# W0212: protected-access


input_apple_jpg = "/home/workspace/mindspore_dataset/910B_dvpp/apple.jpg"
result_data_dir = "/home/workspace/mindspore_dataset/910B_dvpp/testAscend910BDvpp"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_decode_dvpp():
    """
    Feature: Decode op when Ascend910B
    Description: Test eager support for Decode with Dvpp
    Expectation: Output image info from op is correct
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    # read file in binary mode
    img_bytes = np.fromfile(input_apple_jpg, dtype=np.uint8)
    img_decode = vision.Decode().device("Ascend")(img_bytes)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img_decode), img_decode.shape))

    # the check file
    check_img = cv2.imread(os.path.join(result_data_dir, "apple_dvpp_decode.png"))
    check_img = cv2.cvtColor(check_img, cv2.COLOR_BGR2RGB)

    assert img_decode.shape == (2268, 4032, 3)
    assert img_decode.dtype == np.uint8
    assert (img_decode == check_img).all()

    # decode + resize
    img_decode2 = vision.Decode().device("Ascend")(img_bytes)
    img_resize = vision.Resize(size=(64, 32)).device("Ascend")(img_decode2)

    assert img_resize.shape == (64, 32, 3)
    assert img_resize.dtype == np.uint8

    # check the result
    check_img = cv2.imread(os.path.join(result_data_dir, "apple_dvpp_resize.png"))
    check_img = cv2.cvtColor(check_img, cv2.COLOR_BGR2RGB)
    assert (img_resize == check_img).all()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_decode_dvpp_exception():
    """
    Feature: Decode op when Ascend910B with exception
    Description: Test eager support for Decode with Dvpp
    Expectation: Output image info from op is correct
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    # bmp
    img_bytes = np.fromfile(os.path.join(result_data_dir, "apple.bmp"), dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Decode().device("Ascend")(img_bytes)
    assert "Invalid image type. Currently only support JPG." in str(error_info.value)

    # png
    img_bytes = np.fromfile(os.path.join(result_data_dir, "apple.png"), dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Decode().device("Ascend")(img_bytes)
    assert "Invalid image type. Currently only support JPG." in str(error_info.value)

    # dtype is float
    img_bytes = np.fromfile(input_apple_jpg)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Decode().device("Ascend")(img_bytes)
    assert "Invalid data type. Currently only support uint8." in str(error_info.value)

    # not 1D
    img_hw = np.ones((224, 224), dtype=np.uint8)
    with pytest.raises(TypeError) as error_info:
        _ = vision.Decode().device("Ascend")(img_hw)
    assert "The number of array dimensions of the encoded image should be 1" in str(error_info.value)

    # invalid device_target
    img_bytes = np.fromfile(input_apple_jpg, dtype=np.uint8)
    with pytest.raises(ValueError) as error_info:
        _ = vision.Decode().device("CPUS")(img_bytes)
    assert "Input device_target is not within the valid set of ['CPU', 'Ascend']" in str(error_info.value)

    with pytest.raises(TypeError) as error_info:
        _ = vision.Decode().device(123)(img_bytes)
    assert "Argument device_target with value 123 is not of type [<class 'str'>]," in str(error_info.value)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_resize_dvpp():
    """
    Feature: Resize op when Ascend910B
    Description: Test eager support for Resize with Dvpp
    Expectation: Output image info from op is correct
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    # HWC
    img = cv2.imread(input_apple_jpg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img_resize = vision.Resize(size=(64, 32)).device("Ascend")(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img_resize), img_resize.shape))

    assert img_resize.shape == (64, 32, 3)

    # check the result
    check_img = cv2.imread(os.path.join(result_data_dir, "apple_dvpp_resize.png"))
    check_img = cv2.cvtColor(check_img, cv2.COLOR_BGR2RGB)
    assert (img_resize == check_img).all()

    # interpolation is BILINEAR
    img_resize1 = vision.Resize(size=(64, 32), interpolation=vision.Inter.BILINEAR).device("Ascend")(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img_resize1), img_resize1.shape))
    assert img_resize1.shape == (64, 32, 3)

    # check the result
    check_img1 = cv2.imread(os.path.join(result_data_dir, "apple_dvpp_resize_BILINEAR.png"))
    check_img1 = cv2.cvtColor(check_img1, cv2.COLOR_BGR2RGB)
    assert (img_resize1 == check_img1).all()

    # interpolation is NEAREST
    img_resize2 = vision.Resize(size=(64, 32), interpolation=vision.Inter.NEAREST).device("Ascend")(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img_resize2), img_resize2.shape))
    assert img_resize2.shape == (64, 32, 3)

    # check the result
    check_img2 = cv2.imread(os.path.join(result_data_dir, "apple_dvpp_resize_NEAREST.png"))
    check_img2 = cv2.cvtColor(check_img2, cv2.COLOR_BGR2RGB)
    assert (img_resize2 == check_img2).all()

    # interpolation is CUBIC
    img_resize3 = vision.Resize(size=(64, 32), interpolation=vision.Inter.CUBIC).device("Ascend")(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img_resize3), img_resize3.shape))
    assert img_resize3.shape == (64, 32, 3)

    # check the result
    check_img3 = cv2.imread(os.path.join(result_data_dir, "apple_dvpp_resize_CUBIC.png"))
    check_img3 = cv2.cvtColor(check_img3, cv2.COLOR_BGR2RGB)
    assert (img_resize3 == check_img3).all()

    # interpolation is BICUBIC
    img_resize4 = vision.Resize(size=(64, 32), interpolation=vision.Inter.BICUBIC).device("Ascend")(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img_resize4), img_resize4.shape))
    assert img_resize4.shape == (64, 32, 3)

    # check the result
    check_img4 = cv2.imread(os.path.join(result_data_dir, "apple_dvpp_resize_BICUBIC.png"))
    check_img4 = cv2.cvtColor(check_img4, cv2.COLOR_BGR2RGB)
    assert (img_resize4 == check_img4).all()

    # the input is HW
    img = np.ones([224, 224], dtype=np.uint8)
    img_resize_hw = vision.Resize(size=(64, 32)).device("Ascend")(img)
    assert img_resize_hw.shape == (64, 32)

    # the input is HW1
    img = np.ones([224, 224, 1], dtype=np.uint8)
    img_resize_hw1 = vision.Resize(size=(64, 32)).device("Ascend")(img)
    assert img_resize_hw1.shape == (64, 32)

    # the input is 1HW1
    img = np.ones([1, 224, 224, 1], dtype=np.uint8)
    img_resize_1hw1 = vision.Resize(size=(64, 32)).device("Ascend")(img)
    assert img_resize_1hw1.shape == (64, 32)

    # the input is float HW3
    img = np.ones([224, 224, 3], dtype=np.float32)
    img_resize_float_hw3 = vision.Resize(size=(64, 32)).device("Ascend")(img)
    assert img_resize_float_hw3.shape == (64, 32, 3)
    assert img_resize_float_hw3.dtype == np.float32


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_resize_dvpp_exception():
    """
    Feature: Resize op when Ascend910B
    Description: Test eager support for Resize with Dvpp when invalid input
    Expectation: Success
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    f = open(input_apple_jpg, "rb")
    img = f.read()
    f.close()

    # the input is bytes
    with pytest.raises(TypeError) as error_info:
        img = vision.Resize(size=(64, 32)).device("Ascend")(img)
    assert "Input should be NumPy or PIL image" in str(error_info.value)

    # the input is list
    img = np.ones([1024], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Resize(size=(64, 32)).device("Ascend")(img)
    assert "the input tensor is not HW, HWC or 1HWC." in str(error_info.value)

    # the input is HW2
    img = np.ones([224, 224, 2], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Resize(size=(64, 32)).device("Ascend")(img)
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(error_info.value)

    # the input is HW4
    img = np.ones([224, 224, 4], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Resize(size=(64, 32)).device("Ascend")(img)
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(error_info.value)

    # the input is 23HW3
    img = np.ones([2, 3, 224, 224, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Resize(size=(64, 32)).device("Ascend")(img)
    assert "The input tensor is not of shape [H,W], [H,W,C] or [N,H,W,C]." in str(error_info.value)

    # the input is 3HW1
    img = np.ones([3, 224, 224, 1], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Resize(size=(64, 32)).device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    # the input is 3HW3
    img = np.ones([3, 224, 224, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Resize(size=(64, 32)).device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    # the input is 6HW3
    img = np.ones([6, 224, 224, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Resize(size=(64, 32)).device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    # the input is float 9HW3
    img = np.ones([9, 224, 224, 3], dtype=np.float32)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Resize(size=(64, 32)).device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    img = cv2.imread(input_apple_jpg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # the interpolation is invalid
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Resize(size=(64, 32), interpolation=vision.Inter.AREA).device("Ascend")(img)
    assert "Invalid interpolation mode, only support BILINEAR, CUBIC and NEAREST" in str(error_info.value)

    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Resize(size=(64, 32), interpolation=vision.Inter.PILCUBIC).device("Ascend")(img)
    assert "Invalid interpolation mode, only support BILINEAR, CUBIC and NEAREST" in str(error_info.value)

    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Resize(size=(64, 32), interpolation=vision.Inter.ANTIALIAS).device("Ascend")(img)
    assert "Invalid interpolation mode, only support BILINEAR, CUBIC and NEAREST" in str(error_info.value)

    # the device(device_target) is invalid
    with pytest.raises(TypeError) as error_info:
        _ = vision.Resize(size=(64, 32)).device(12)
    assert "Argument device_target with value 12 is not of type [<class 'str'>]" in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        _ = vision.Resize(size=(64, 32)).device("Asscend")
    assert "Input device_target is not within the valid set of ['CPU', 'Ascend']" in str(error_info.value)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_eager_resize_dvpp_exception_with_910A():
    """
    Feature: Resize op when Ascend910A
    Description: Will prompt exception not supported
    Expectation: With exception
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    # HWC
    img = cv2.imread(input_apple_jpg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    # run the op
    with pytest.raises(RuntimeError):
        _ = vision.Resize(size=(64, 32)).device("Ascend")(img)

    # retry to run the op
    with pytest.raises(RuntimeError):
        _ = vision.Resize(size=(64, 32)).device("Ascend")(img)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_normalize_dvpp():
    """
    Feature: Normalize op when Ascend910B
    Description: Test eager support for Normalize with Dvpp
    Expectation: Output image info from op is correct
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    # HWC
    img = cv2.imread(input_apple_jpg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_resize = vision.Resize(size=(64, 32))(img)
    mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
    std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
    img_normalize = vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_resize)
    assert img_normalize.shape == (64, 32, 3)
    assert img_normalize.dtype == np.float32

    check_result = np.load(os.path.join(result_data_dir, 'apple_dvpp_normalize.npy'))
    assert (check_result == img_normalize).all()

    # 1HWC
    img = cv2.imread(input_apple_jpg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_resize = vision.Resize(size=(64, 32))(img)
    img_resize = np.expand_dims(img_resize, axis=0)
    assert img_resize.shape == (1, 64, 32, 3)
    mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
    std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
    img_normalize = vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_resize)
    assert img_normalize.shape == (64, 32, 3)
    assert img_normalize.dtype == np.float32

    check_result = np.load(os.path.join(result_data_dir, 'apple_dvpp_normalize.npy'))
    assert (check_result == img_normalize).all()

    # CHW
    img = cv2.imread(input_apple_jpg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_resize = vision.Resize(size=(64, 32))(img)
    img_resize = np.transpose(img_resize, (2, 0, 1))
    assert img_resize.shape == (3, 64, 32)
    mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
    std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
    img_normalize = vision.Normalize(mean=mean_vec, std=std_vec, is_hwc=False).device("Ascend")(img_resize)
    assert img_normalize.shape == (3, 64, 32)
    assert img_normalize.dtype == np.float32

    check_result = np.load(os.path.join(result_data_dir, 'apple_dvpp_normalize.npy'))
    check_result = np.transpose(check_result, (2, 0, 1))
    assert (check_result == img_normalize).all()

    # 1CHW
    img = cv2.imread(input_apple_jpg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_resize = vision.Resize(size=(64, 32))(img)
    img_resize = np.transpose(img_resize, (2, 0, 1))
    img_resize = np.expand_dims(img_resize, axis=0)
    assert img_resize.shape == (1, 3, 64, 32)
    mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
    std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
    img_normalize = vision.Normalize(mean=mean_vec, std=std_vec, is_hwc=False).device("Ascend")(img_resize)
    assert img_normalize.shape == (3, 64, 32)
    assert img_normalize.dtype == np.float32

    check_result = np.load(os.path.join(result_data_dir, 'apple_dvpp_normalize.npy'))
    check_result = np.transpose(check_result, (2, 0, 1))
    assert (check_result == img_normalize).all()

    # HW
    img = np.ones([224, 224], dtype=np.uint8) * 221
    mean_vec = [0.475 * 255]
    std_vec = [0.275 * 255]
    img_normalize = vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img)
    assert img_normalize.shape == (224, 224)
    assert img_normalize.dtype == np.float32

    check_result = np.load(os.path.join(result_data_dir, 'apple_dvpp_normalize_hw.npy'))
    assert (check_result == img_normalize).all()

    # HW1
    img = np.ones([224, 224], dtype=np.uint8) * 221
    mean_vec = [0.475 * 255]
    std_vec = [0.275 * 255]
    img = np.expand_dims(img, axis=2)
    assert img.shape == (224, 224, 1)
    img_normalize = vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img)
    assert img_normalize.shape == (224, 224)
    assert img_normalize.dtype == np.float32

    check_result = np.load(os.path.join(result_data_dir, 'apple_dvpp_normalize_hw.npy'))
    assert (check_result == img_normalize).all()

    # 1HW
    img = np.ones([224, 224], dtype=np.uint8) * 221
    mean_vec = [0.475 * 255]
    std_vec = [0.275 * 255]
    img = np.expand_dims(img, axis=0)
    assert img.shape == (1, 224, 224)
    img_normalize = vision.Normalize(mean=mean_vec, std=std_vec, is_hwc=False).device("Ascend")(img)
    assert img_normalize.shape == (224, 224)
    assert img_normalize.dtype == np.float32

    check_result = np.load(os.path.join(result_data_dir, 'apple_dvpp_normalize_hw.npy'))
    assert (check_result == img_normalize).all()

    # 1HW1
    img = np.ones([224, 224], dtype=np.uint8) * 221
    mean_vec = [0.475 * 255]
    std_vec = [0.275 * 255]
    img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0)
    assert img.shape == (1, 224, 224, 1)
    img_normalize = vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img)
    assert img_normalize.shape == (224, 224)
    assert img_normalize.dtype == np.float32

    check_result = np.load(os.path.join(result_data_dir, 'apple_dvpp_normalize_hw.npy'))
    assert (check_result == img_normalize).all()

    # float HWC
    img = cv2.imread(input_apple_jpg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_resize = vision.Resize(size=(64, 32))(img)
    assert img_resize.dtype == np.uint8
    img_resize = img_resize.astype(np.float32)
    assert img_resize.dtype == np.float32
    mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
    std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
    img_normalize = vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_resize)
    assert img_normalize.shape == (64, 32, 3)
    assert img_normalize.dtype == np.float32

    check_result = np.load(os.path.join(result_data_dir, 'apple_dvpp_normalize.npy'))
    assert (check_result == img_normalize).all()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_normalize_dvpp_exception():
    """
    Feature: Normalize op when Ascend910B with exception
    Description: Test eager support for Normalize with Dvpp
    Expectation: Output image info from op is correct
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    # mean size is 3, input is HW
    mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
    std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
    img = np.ones([224, 224], dtype=np.uint8) * 221
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img)
    assert "The channel is not equal to the size of mean or std." in str(error_info.value)

    # mean size is 2, input is HWC
    mean_vec = [0.475 * 255, 0.451 * 255]
    std_vec = [0.275 * 255, 0.267 * 255]
    img = np.ones([224, 224, 3], dtype=np.uint8) * 221
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img)
    assert "The channel is not equal to the size of mean or std." in str(error_info.value)

    # mean size is 2, input is CHW
    mean_vec = [0.475 * 255, 0.451 * 255]
    std_vec = [0.275 * 255, 0.267 * 255]
    img = np.ones([3, 224, 224], dtype=np.uint8) * 221
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Normalize(mean=mean_vec, std=std_vec, is_hwc=False).device("Ascend")(img)
    assert "The channel is not equal to the size of mean or std." in str(error_info.value)

    # HW3, but is_hwc=False
    img = np.ones([224, 224, 3], dtype=np.uint8) * 221
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Normalize(mean=mean_vec, std=std_vec, is_hwc=False).device("Ascend")(img)
    assert "The channel of the input tensor of shape [C,H,W] is not 1 or 3" in str(error_info.value)

    # 3HW, but is_hwc=True
    img = np.ones([3, 224, 224], dtype=np.uint8) * 221
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Normalize(mean=mean_vec, std=std_vec, is_hwc=True).device("Ascend")(img)
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(error_info.value)

    # HW2
    img = np.ones([224, 224, 2], dtype=np.uint8) * 221
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img)
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(error_info.value)

    # 4HW
    img = np.ones([4, 224, 224], dtype=np.uint8) * 221
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Normalize(mean=mean_vec, std=std_vec, is_hwc=False).device("Ascend")(img)
    assert "The channel of the input tensor of shape [C,H,W] is not 1 or 3" in str(error_info.value)

    # 2HWC
    mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
    std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
    img = np.ones([2, 224, 224, 3], dtype=np.uint8) * 221
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    # 3CHW
    mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
    std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
    img = np.ones([3, 3, 224, 224], dtype=np.uint8) * 221
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Normalize(mean=mean_vec, std=std_vec, is_hwc=False).device("Ascend")(img)
    assert "The input tensor NCHW should be 1CHW or CHW." in str(error_info.value)

    # float16
    img = np.ones([224, 224, 3], dtype=np.float16) * 221
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img)
    assert "The input data is not uint8 or float32." in str(error_info.value)

    # int32
    img = np.ones([3, 224, 224], dtype=np.float16) * 221
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Normalize(mean=mean_vec, std=std_vec, is_hwc=False).device("Ascend")(img)
    assert "The input data is not uint8 or float32." in str(error_info.value)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_multi_dvpp_op_dvpp_cpu_dvpp():
    """
    Feature: Multi ops when Ascend910B with global executor
    Description: Test eager support for multi op with Dvpp
    Expectation: Output image info from op is correct
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    # decode(dvpp)
    img_bytes = np.fromfile(input_apple_jpg, dtype=np.uint8)
    img_decode = vision.Decode().device("Ascend")(img_bytes)

    assert img_decode.shape == (2268, 4032, 3)
    assert img_decode.dtype == np.uint8

    # resize(cpu)
    img_resize = vision.Resize(size=(64, 32))(img_decode)

    assert img_resize.shape == (64, 32, 3)
    assert img_resize.dtype == np.uint8

    # normalize(dvpp)
    mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
    std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
    img_normalize = vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_resize)
    assert img_normalize.shape == (64, 32, 3)
    assert img_normalize.dtype == np.float32


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_multi_dvpp_op_dvpp_dvpp_cpu():
    """
    Feature: Multi ops when Ascend910B with global executor
    Description: Test eager support for multi op with Dvpp
    Expectation: Output image info from op is correct
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    # decode(dvpp)
    img_bytes = np.fromfile(input_apple_jpg, dtype=np.uint8)
    img_decode = vision.Decode().device("Ascend")(img_bytes)

    assert img_decode.shape == (2268, 4032, 3)
    assert img_decode.dtype == np.uint8

    # resize(dvpp)
    img_resize = vision.Resize(size=(64, 32)).device("Ascend")(img_decode)

    assert img_resize.shape == (64, 32, 3)
    assert img_resize.dtype == np.uint8

    # normalize(cpu)
    mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
    std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
    img_normalize = vision.Normalize(mean=mean_vec, std=std_vec)(img_resize)
    assert img_normalize.shape == (64, 32, 3)
    assert img_normalize.dtype == np.float32


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_multi_dvpp_op_cpu_dvpp_dvpp():
    """
    Feature: Multi ops when Ascend910B with global executor
    Description: Test eager support for multi op with Dvpp
    Expectation: Output image info from op is correct
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    # decode(cpu)
    img_bytes = np.fromfile(input_apple_jpg, dtype=np.uint8)
    img_decode = vision.Decode()(img_bytes)

    assert img_decode.shape == (2268, 4032, 3)
    assert img_decode.dtype == np.uint8

    # resize(dvpp)
    img_resize = vision.Resize(size=(64, 32)).device("Ascend")(img_decode)

    assert img_resize.shape == (64, 32, 3)
    assert img_resize.dtype == np.uint8

    # normalize(dvpp)
    mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
    std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
    img_normalize = vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_resize)
    assert img_normalize.shape == (64, 32, 3)
    assert img_normalize.dtype == np.float32


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_horizontal_flip_dvpp():
    """
    Feature: Horizontal Flip op when Ascend910B
    Description: Test eager support for Horizontal Filp with Dvpp
    Expectation: Output image info from op is correct
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    # HWC
    img = cv2.imread(input_apple_jpg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img_horizontal_flip = vision.HorizontalFlip().device("Ascend")(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img_horizontal_flip), img_horizontal_flip.shape))

    assert img_horizontal_flip.shape == (2268, 4032, 3)

    # check the result
    check_img = cv2.imread(os.path.join(result_data_dir, "dvpp_horizontal_flip.png"))
    check_img = cv2.cvtColor(check_img, cv2.COLOR_BGR2RGB)
    assert (img_horizontal_flip == check_img).all()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_horizontal_flip_dvpp_exception():
    """
    Feature: Horizontal Flip op when Ascend910B
    Description: Test eager support for horizontal flip with Dvpp when invalid input
    Expectation: Success
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    f = open(input_apple_jpg, "rb")
    img = f.read()
    f.close()

    # the input is list
    img = np.ones([1024], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.HorizontalFlip().device("Ascend")(img)
    assert "invalid input shape, only support NHWC input" in str(error_info.value)

    # the input is HW2
    img = np.ones([224, 224, 2], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.HorizontalFlip().device("Ascend")(img)
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(error_info.value)

    # the input is HW4
    img = np.ones([224, 224, 4], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.HorizontalFlip().device("Ascend")(img)
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(error_info.value)

    # the input is 23HW3
    img = np.ones([2, 3, 224, 224, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.HorizontalFlip().device("Ascend")(img)
    assert "The input tensor is not of shape [H,W], [H,W,C] or [N,H,W,C]." in str(error_info.value)

    # the input is 3HW1
    img = np.ones([3, 224, 224, 1], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.HorizontalFlip().device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    # the input is 3HW3
    img = np.ones([3, 224, 224, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.HorizontalFlip().device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    # the input is 6HW3
    img = np.ones([6, 224, 224, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.HorizontalFlip().device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    # the input is float 9HW3
    img = np.ones([9, 224, 224, 3], dtype=np.float32)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.HorizontalFlip().device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    img = cv2.imread(input_apple_jpg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # the device(device_target) is invalid
    with pytest.raises(TypeError) as error_info:
        _ = vision.HorizontalFlip().device(20)
    assert "Argument device_target with value 20 is not of type [<class 'str'>]" in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        _ = vision.HorizontalFlip().device("Asscend")
    assert "Input device_target is not within the valid set of ['CPU', 'Ascend']" in str(error_info.value)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_vertical_flip_dvpp():
    """
    Feature: Vertical Flip op when Ascend910B
    Description: Test eager support for Vertical Filp with Dvpp
    Expectation: Output image info from op is correct
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    # HWC
    img = cv2.imread(input_apple_jpg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img_vertical_flip = vision.VerticalFlip().device("Ascend")(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img_vertical_flip), img_vertical_flip.shape))

    assert img_vertical_flip.shape == (2268, 4032, 3)

    # check the result
    check_img = cv2.imread(os.path.join(result_data_dir, "dvpp_vertical_flip.png"))
    check_img = cv2.cvtColor(check_img, cv2.COLOR_BGR2RGB)
    assert (img_vertical_flip == check_img).all()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_vertical_flip_dvpp_exception():
    """
    Feature: Vertical Flip op when Ascend910B
    Description: Test eager support for vertical flip with Dvpp when invalid input
    Expectation: Success
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    f = open(input_apple_jpg, "rb")
    img = f.read()
    f.close()

    # the input is list
    img = np.ones([1024], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.VerticalFlip().device("Ascend")(img)
    assert "invalid input shape, only support NHWC input" in str(error_info.value)

    # the input is HW2
    img = np.ones([224, 224, 2], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.VerticalFlip().device("Ascend")(img)
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(error_info.value)

    # the input is HW4
    img = np.ones([224, 224, 4], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.VerticalFlip().device("Ascend")(img)
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(error_info.value)

    # the input is 23HW3
    img = np.ones([2, 3, 224, 224, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.VerticalFlip().device("Ascend")(img)
    assert "The input tensor is not of shape [H,W], [H,W,C] or [N,H,W,C]." in str(error_info.value)

    # the input is 3HW1
    img = np.ones([3, 224, 224, 1], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.VerticalFlip().device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    # the input is 3HW3
    img = np.ones([3, 224, 224, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.VerticalFlip().device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    # the input is 6HW3
    img = np.ones([6, 224, 224, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.VerticalFlip().device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    # the input is float 9HW3
    img = np.ones([9, 224, 224, 3], dtype=np.float32)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.VerticalFlip().device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    img = cv2.imread(input_apple_jpg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # the device(device_target) is invalid
    with pytest.raises(TypeError) as error_info:
        _ = vision.VerticalFlip().device(20)
    assert "Argument device_target with value 20 is not of type [<class 'str'>]" in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        _ = vision.VerticalFlip().device("Asscend")
    assert "Input device_target is not within the valid set of ['CPU', 'Ascend']" in str(error_info.value)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_resize_crop_dvpp():
    """
    Feature: Resize crop op when Ascend910B
    Description: Test eager support for Resize crop with Dvpp
    Expectation: Output image info from op is correct
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    # HWC
    img = cv2.imread(input_apple_jpg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img_resize_crop = vision.ResizedCrop(0, 0, 2000, 3000, (500, 500)).device("Ascend")(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img_resize_crop), img_resize_crop.shape))

    assert img_resize_crop.shape == (500, 500, 3)

    # check the result
    check_img = cv2.imread(os.path.join(result_data_dir, "dvpp_crop_resize.png"))
    check_img = cv2.cvtColor(check_img, cv2.COLOR_BGR2RGB)
    assert (img_resize_crop == check_img).all()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_resize_crop_dvpp_exception():
    """
    Feature: Resize crop op when Ascend910B
    Description: Test eager support for resize crop with Dvpp when invalid input
    Expectation: Success
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    f = open(input_apple_jpg, "rb")
    img = f.read()
    f.close()

    # the input is HW2
    img = np.ones([224, 224, 2], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.ResizedCrop(0, 0, 128, 128, (100, 75)).device("Ascend")(img)
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(error_info.value)

    # the input is HW4
    img = np.ones([224, 224, 4], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.ResizedCrop(0, 0, 128, 128, (100, 75)).device("Ascend")(img)
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(error_info.value)

    # the input is 23HW3
    img = np.ones([2, 3, 224, 224, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.ResizedCrop(0, 0, 128, 128, (100, 75)).device("Ascend")(img)
    assert "The input tensor is not of shape [H,W], [H,W,C] or [N,H,W,C]." in str(error_info.value)

    # the input is 3HW1
    img = np.ones([3, 224, 224, 1], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.ResizedCrop(0, 0, 128, 128, (100, 75)).device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    # the input is 3HW3
    img = np.ones([3, 224, 224, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.ResizedCrop(0, 0, 128, 128, (100, 75)).device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    # the input is 6HW3
    img = np.ones([6, 224, 224, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.ResizedCrop(0, 0, 128, 128, (100, 75)).device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    # the input is float 9HW3
    img = np.ones([9, 224, 224, 3], dtype=np.float32)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.ResizedCrop(0, 0, 128, 128, (100, 75)).device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    img = cv2.imread(input_apple_jpg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # the device(device_target) is invalid
    with pytest.raises(TypeError) as error_info:
        _ = vision.ResizedCrop(0, 0, 128, 128, (100, 75)).device(20)
    assert "Argument device_target with value 20 is not of type [<class 'str'>]" in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        _ = vision.ResizedCrop(0, 0, 128, 128, (100, 75)).device("Asscend")
    assert "Input device_target is not within the valid set of ['CPU', 'Ascend']" in str(error_info.value)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_perspective_dvpp():
    """
    Feature: Perspective op when Ascend910B
    Description: Test eager support for Perspective with Dvpp
    Expectation: Output image info from op is correct
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    start_points = [[0, 63], [63, 63], [63, 0], [0, 0]]
    end_points = [[0, 32], [32, 32], [32, 0], [0, 0]]

    # HWC
    img = cv2.imread(input_apple_jpg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img_perspective = vision.Perspective(start_points, end_points).device("Ascend")(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img_perspective), img_perspective.shape))

    assert img_perspective.shape == (2268, 4032, 3)

    # check the result
    check_img = cv2.imread(os.path.join(result_data_dir, "dvpp_perspective.png"))
    check_img = cv2.cvtColor(check_img, cv2.COLOR_BGR2RGB)
    assert (img_perspective == check_img).all()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_perspective_dvpp_exception():
    """
    Feature: Perspective op when Ascend910B
    Description: Test eager support for perspective with Dvpp when invalid input
    Expectation: Success
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    f = open(input_apple_jpg, "rb")
    img = f.read()
    f.close()

    start_points = [[0, 63], [63, 63], [63, 0], [0, 0]]
    end_points = [[0, 32], [32, 32], [32, 0], [0, 0]]

    # the input is list
    img = np.ones([1024], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Perspective(start_points, end_points).device("Ascend")(img)
    assert "invalid input shape, only support NHWC input" in str(error_info.value)

    # the input is HW2
    img = np.ones([224, 224, 2], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Perspective(start_points, end_points).device("Ascend")(img)
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(error_info.value)

    # the input is HW4
    img = np.ones([224, 224, 4], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Perspective(start_points, end_points).device("Ascend")(img)
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(error_info.value)

    # the input is 23HW3
    img = np.ones([2, 3, 224, 224, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Perspective(start_points, end_points).device("Ascend")(img)
    assert "The input tensor is not of shape [H,W], [H,W,C] or [N,H,W,C]." in str(error_info.value)

    # the input is 3HW1
    img = np.ones([3, 224, 224, 1], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Perspective(start_points, end_points).device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    # the input is 3HW3
    img = np.ones([3, 224, 224, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Perspective(start_points, end_points).device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    # the input is 6HW3
    img = np.ones([6, 224, 224, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Perspective(start_points, end_points).device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    # the input is float 9HW3
    img = np.ones([9, 224, 224, 3], dtype=np.float32)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Perspective(start_points, end_points).device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    img = cv2.imread(input_apple_jpg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # the device(device_target) is invalid
    with pytest.raises(TypeError) as error_info:
        _ = vision.Perspective(start_points, end_points).device(20)
    assert "Argument device_target with value 20 is not of type [<class 'str'>]" in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        _ = vision.Perspective(start_points, end_points).device("Asscend")
    assert "Input device_target is not within the valid set of ['CPU', 'Ascend']" in str(error_info.value)

    image = Image.open(input_apple_jpg)
    start_points = [[0, 63], [63, 63], [63, 0], [0, 0]]
    end_points = [[0, 32], [32, 32], [32, 0], [0, 0]]

    with pytest.raises(TypeError) as error_info:
        _ = vision.Perspective(start_points, end_points).device("Ascend")(image)
    assert "The input PIL Image cannot be executed on Ascend, " in str(error_info.value)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_compose_dvpp_ops():
    """
    Feature: Compose multi dvpp ops on Ascend910B
    Description: Test composing multi DVPP ops in eager mode
    Expectation: Output image info from op is correct
    """
    ms.set_context(device_target="Ascend")

    # dvpp decode + resize + normaize eager
    img_bytes = np.fromfile(input_apple_jpg, dtype=np.uint8)
    mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
    std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
    decode_resize_normalize_compose_dvpp = \
        transforms.Compose([vision.Decode().device("Ascend"),
                            vision.Resize(224).device("Ascend"),
                            vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend")])
    image_normalize = decode_resize_normalize_compose_dvpp(img_bytes)
    assert image_normalize.shape == (224, 398, 3)
    assert image_normalize.dtype == np.float32

    # don't support mix usage
    with pytest.raises(RuntimeError) as error_info:
        decode_resize_normalize_compose_dvpp_mixed = \
            transforms.Compose([vision.Decode(),
                                vision.Resize(224).device("Ascend"),
                                vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend")])
        image_normalize = decode_resize_normalize_compose_dvpp_mixed(img_bytes)
    assert "Building Transform ops failed!" in str(error_info.value)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_crop_dvpp():
    """
    Feature: Crop op on Ascend910B
    Description: Test eager support for Crop with Dvpp
    Expectation: Output image info from op is correct
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    # HWC input
    img = cv2.imread(input_apple_jpg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img_transformed = vision.Crop((1000, 2000), (400, 500)).device("Ascend")(img)
    logger.info("Image.type: {}, Image.shape: {}".format(img_transformed.dtype, img_transformed.shape))

    assert img_transformed.shape == (400, 500, 3)

    # check the result
    check_img = cv2.imread(input_apple_jpg)
    check_img = cv2.cvtColor(check_img, cv2.COLOR_BGR2RGB)
    check_img = vision.Crop((1000, 2000), (400, 500)).device("CPU")(check_img)
    assert (img_transformed == check_img).all()

    # HW input
    img = np.ones((300, 400)).astype(np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img_transformed = vision.Crop((100, 200), (40, 50)).device("Ascend")(img)
    logger.info("Image.type: {}, Image.shape: {}".format(img_transformed.dtype, img_transformed.shape))

    assert img_transformed.shape == (40, 50,)

    # check the result
    check_img = np.ones((300, 400)).astype(np.uint8)
    check_img = vision.Crop((100, 200), (40, 50)).device("CPU")(check_img)

    assert (img_transformed == check_img).all()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_crop_dvpp_exception():
    """
    Feature: Crop op on Ascend910B
    Description: Test eager support for crop with Dvpp when invalid input
    Expectation: Success
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    # the input is list
    img = np.ones([1024], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Crop((0, 0), 250).device("Ascend")(img)
    assert "the input tensor is not HW, HWC or 1HWC," in str(error_info.value)

    # the input is HW2
    img = np.ones([224, 224, 2], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Crop((0, 0), 250).device("Ascend")(img)
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(error_info.value)

    # the input is 3HW1
    img = np.ones([3, 224, 224, 1], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Crop((0, 0), 250).device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    # the input is 23HW3
    img = np.ones([2, 3, 224, 224, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Crop((0, 0), 250).device("Ascend")(img)
    assert "The input tensor is not of shape [H,W], [H,W,C] or [N,H,W,C]." in str(error_info.value)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_pad_dvpp():
    """
    Feature: Pad op on Ascend910B
    Description: Test eager support for Pad with Dvpp
    Expectation: Output image info from op is correct
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    # HWC input
    img = cv2.imread(input_apple_jpg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img_transformed = vision.Pad([10, 20, 30, 40]).device("Ascend")(img)
    logger.info("Image.type: {}, Image.shape: {}".format(img_transformed.dtype, img_transformed.shape))

    assert img_transformed.shape == (2328, 4072, 3)

    # check the result
    check_img = cv2.imread(input_apple_jpg)
    check_img = cv2.cvtColor(check_img, cv2.COLOR_BGR2RGB)
    check_img = vision.Pad([10, 20, 30, 40]).device("CPU")(check_img)
    assert (img_transformed == check_img).all()

    # HW input
    img = np.ones((300, 400)).astype(np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img_transformed = vision.Pad([10, 20]).device("Ascend")(img)
    logger.info("Image.type: {}, Image.shape: {}".format(img_transformed.dtype, img_transformed.shape))

    assert img_transformed.shape == (340, 420)

    # check the result
    check_img = np.ones((300, 400)).astype(np.uint8)
    check_img = vision.Pad([10, 20]).device("CPU")(check_img)

    assert (img_transformed == check_img).all()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_pad_dvpp_exception():
    """
    Feature: Pad op on Ascend910B
    Description: Test eager support for Pad with Dvpp when invalid input
    Expectation: Success
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    # the input is list
    img = np.ones([1024], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Pad([20, 30]).device("Ascend")(img)
    assert "the input tensor is not HW, HWC or 1HWC," in str(error_info.value)

    # the input is HW2
    img = np.ones([224, 224, 2], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Pad([20, 30]).device("Ascend")(img)
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(error_info.value)

    # the input is 3HW1
    img = np.ones([3, 224, 224, 1], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Pad([20, 30]).device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    # the input is 23HW3
    img = np.ones([2, 3, 224, 224, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Pad([20, 30]).device("Ascend")(img)
    assert "The input tensor is not of shape [H,W], [H,W,C] or [N,H,W,C]." in str(error_info.value)

    # the input is out of [4, 6] to [32768, 32768]
    img = np.ones([3, 6, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Pad([10, 10]).device("Ascend")(img)
    assert "the input shape should be from [4, 6] to [32768, 32768]" in str(error_info.value)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_gaussian_blur_dvpp():
    """
    Feature: Gaussian blur op on Ascend910B
    Description: Test eager support for gaussian blur with Dvpp
    Expectation: Output image info from op is correct
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    # HWC input
    img = cv2.imread(input_apple_jpg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img_transformed = vision.GaussianBlur(5, 5).device("Ascend")(img)
    logger.info("Image.type: {}, Image.shape: {}".format(img_transformed.dtype, img_transformed.shape))

    assert img_transformed.shape == (2268, 4032, 3)

    # HW input
    img = np.ones((300, 400)).astype(np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img_transformed = vision.GaussianBlur(3, 3).device("Ascend")(img)
    logger.info("Image.type: {}, Image.shape: {}".format(img_transformed.dtype, img_transformed.shape))

    assert img_transformed.shape == (300, 400)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_gaussian_blur_dvpp_exception():
    """
    Feature: Gaussian blur op on Ascend910B
    Description: Test eager support for Gaussian blur with Dvpp when invalid input
    Expectation: Success
    """
    ms.set_context(device_target="Ascend")

    # the input is list
    img = np.ones([1024], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.GaussianBlur(5, 5).device("Ascend")(img)
    assert "the input tensor is not HW, HWC or 1HWC," in str(error_info.value)

    # the input is HW2
    img = np.ones([224, 224, 2], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.GaussianBlur(5, 5).device("Ascend")(img)
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(error_info.value)

    # the input is 3HW1
    img = np.ones([3, 224, 224, 1], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.GaussianBlur(5, 5).device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    # the input is 23HW3
    img = np.ones([2, 3, 224, 224, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.GaussianBlur(5, 5).device("Ascend")(img)
    assert "The input tensor is not of shape [H,W], [H,W,C] or [N,H,W,C]." in str(error_info.value)

    # the input is out of [4, 6] to [8192, 4096]
    img = np.ones([3, 6, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.GaussianBlur(5, 5).device("Ascend")(img)
    assert "the input shape should be from [4, 6] to [8192, 4096]" in str(error_info.value)

    # the input kernel is invalid
    img = np.ones([30, 60, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.GaussianBlur(9, 9).device("Ascend")(img)
    assert "`kernel_size` only supports values 1, 3, and 5" in str(error_info.value)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_affine_dvpp():
    """
    Feature: Affine op on Ascend910B
    Description: Test eager support for affine with Dvpp
    Expectation: Output image info from op is correct
    """
    ms.set_context(device_target="Ascend")

    # HWC input
    img = cv2.imread(input_apple_jpg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img_transformed = vision.Affine(degrees=15, translate=[0.2, 0.2], scale=1.1, shear=[1, 1]).device("Ascend")(img)
    logger.info("Image.type: {}, Image.shape: {}".format(img_transformed.dtype, img_transformed.shape))

    assert img_transformed.shape == (2268, 4032, 3)

    # HW input
    img = np.ones((300, 400)).astype(np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img_transformed = vision.Affine(degrees=15, translate=[0.2, 0.2], scale=1.1, shear=[1, 1]).device("Ascend")(img)
    logger.info("Image.type: {}, Image.shape: {}".format(img_transformed.dtype, img_transformed.shape))

    assert img_transformed.shape == (300, 400)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_affine_dvpp_exception():
    """
    Feature: Affine op on Ascend910B
    Description: Test eager support for Affine with Dvpp when invalid input
    Expectation: Success
    """
    ms.set_context(device_target="Ascend")

    # the input is list
    img = np.ones([1024], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Affine(degrees=15, translate=[0.2, 0.2], scale=1.1, shear=[1, 1]).device("Ascend")(img)
    assert "the input tensor is not HW, HWC or 1HWC," in str(error_info.value)

    # the input is 23HW3
    img = np.ones([2, 3, 224, 224, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Affine(degrees=15, translate=[0.2, 0.2], scale=1.1, shear=[1, 1]).device("Ascend")(img)
    assert "The input tensor is not of shape [H,W], [H,W,C] or [N,H,W,C]." in str(error_info.value)

    # the input is out of [4, 6] to [32768, 32768]
    img = np.ones([4, 5, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Affine(degrees=15, translate=[0.2, 0.2], scale=1.1, shear=[1, 1]).device("Ascend")(img)
    assert "the input shape should be from [4, 6] to [32768, 32768]" in str(error_info.value)

    # the input kernel is invalid
    img = np.ones([30, 60, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Affine(degrees=15, translate=[0.2, 0.2], scale=1.1, shear=[1, 1],
                            resample=vision.Inter.AREA).device("Ascend")(img)
    assert "Invalid interpolation mode, only support BILINEAR and NEAREST" in str(error_info.value)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_equalize_dvpp():
    """
    Feature: Equalize op on Ascend910B
    Description: Test eager support for equalize with Dvpp
    Expectation: Output image info from op is correct
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    # HWC input
    img = cv2.imread(input_apple_jpg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img_transformed = vision.Equalize().device("Ascend")(img)
    logger.info("Image.type: {}, Image.shape: {}".format(img_transformed.dtype, img_transformed.shape))

    assert img_transformed.shape == (2268, 4032, 3)

    # HW input
    img = np.ones((300, 400)).astype(np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img_transformed = vision.Equalize().device("Ascend")(img)
    logger.info("Image.type: {}, Image.shape: {}".format(img_transformed.dtype, img_transformed.shape))

    assert img_transformed.shape == (300, 400)

    # check the result
    check_img = np.ones((300, 400)).astype(np.uint8)
    check_img = vision.Equalize().device("CPU")(check_img)

    assert (img_transformed == check_img).all()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_equalize_dvpp_exception():
    """
    Feature: Equalize op on Ascend910B
    Description: Test eager support for Equalize with Dvpp when invalid input
    Expectation: Success
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    # the input is list
    img = np.ones([1024], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Equalize().device("Ascend")(img)
    assert "the input tensor is not HW, HWC or 1HWC," in str(error_info.value)

    # the input is HW2
    img = np.ones([224, 224, 2], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Equalize().device("Ascend")(img)
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(error_info.value)

    # the input is 23HW3
    img = np.ones([2, 3, 224, 224, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Equalize().device("Ascend")(img)
    assert "The input tensor is not of shape [H,W], [H,W,C] or [N,H,W,C]." in str(error_info.value)

    # the input is 3HW1
    img = np.ones([3, 224, 224, 1], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Equalize().device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    # the input is out of [4, 6] to [8192, 4096]
    img = np.ones([3, 6, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Equalize().device("Ascend")(img)
    assert "the input shape should be from [4, 6] to [8192, 4096]" in str(error_info.value)

    # the device(device_target) is invalid
    with pytest.raises(TypeError) as error_info:
        _ = vision.Equalize().device(123)
    assert "Argument device_target with value 123 is not of type [<class 'str'>]" in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        _ = vision.Equalize().device("Asscend")
    assert "Input device_target is not within the valid set of ['CPU', 'Ascend']" in str(error_info.value)

@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_auto_contrast_dvpp():
    """
    Feature: AutoContrast op on Ascend910B
    Description: Test eager support for autoContrast with Dvpp
    Expectation: Output image info from op is correct
    """
    ms.set_context(device_target="Ascend")

    img = cv2.imread(input_apple_jpg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    img_transformed = vision.AutoContrast(cutoff=10.0, ignore=[10, 20]).device("Ascend")(img)
    logger.info("Image.type: {}, Image.shape: {}".format(img_transformed.dtype, img_transformed.shape))
    check_img = cv2.imread(input_apple_jpg)
    check_img = cv2.cvtColor(check_img, cv2.COLOR_BGR2RGB)
    check_img = vision.AutoContrast(cutoff=10.0, ignore=[10, 20]).device("CPU")(check_img)
    np.testing.assert_allclose(img_transformed, check_img, 0.005, 0.005)

    img = np.random.randint(0, 255, size=(300, 400), dtype=np.uint8)
    img_copy = copy.copy(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    img_transformed = vision.AutoContrast(cutoff=10.0, ignore=[10, 20]).device("Ascend")(img)
    logger.info("Image.type: {}, Image.shape: {}".format(img_transformed.dtype, img_transformed.shape))
    # check the result
    check_img = vision.AutoContrast(cutoff=10.0, ignore=[10, 20]).device("CPU")(img_copy)
    np.testing.assert_allclose(img_transformed, check_img, 0.005, 0.005)

@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_auto_contrast_dvpp_exception():
    """
    Feature: Equalize op on Ascend910B
    Description: Test eager support for Equalize with Dvpp when invalid input
    Expectation: Success
    """
    ms.set_context(device_target="Ascend")
    # the input is list
    img = np.ones([1024], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.AutoContrast().device("Ascend")(img)
    assert "the input tensor is not HW, HWC or 1HWC," in str(error_info.value)

    # the input is HW2
    img = np.ones([224, 224, 2], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.AutoContrast().device("Ascend")(img)
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(error_info.value)

    # the input is 23HW3
    img = np.ones([2, 3, 224, 224, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.AutoContrast().device("Ascend")(img)
    assert "The input tensor is not of shape [H,W], [H,W,C] or [N,H,W,C]." in str(error_info.value)

    # the input is 3HW1
    img = np.ones([3, 224, 224, 1], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.AutoContrast().device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    # the input is out of [4, 6] to [8192, 4096]
    img = np.ones([3, 6, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.AutoContrast().device("Ascend")(img)
    assert "the input shape should be from [4, 6] to [8192, 4096]" in str(error_info.value)

    # the length of ignore should be less or equal to 256
    img = np.random.randint(0, 255, size=(300, 400), dtype=np.uint8)
    ignore = np.random.randint(0, 255, (1000,)).astype(np.uint8).tolist()
    with pytest.raises(RuntimeError) as error_info:
        img = vision.AutoContrast(cutoff=38.653, ignore=ignore).device("Ascend")(img)
    assert "the length of ignore should be less or equal to 256" in str(error_info.value)

    # the device(device_target) is invalid
    with pytest.raises(TypeError) as error_info:
        _ = vision.AutoContrast().device(123)
    assert "Argument device_target with value 123 is not of type [<class 'str'>]" in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        _ = vision.AutoContrast().device("Asscend")
    assert "Input device_target is not within the valid set of ['CPU', 'Ascend']" in str(error_info.value)

@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_adjust_sharpness_dvpp():
    """
    Feature: AdjustSharpness op on Ascend910B
    Description: Test eager support for AdjustSharpness with Dvpp
    Expectation: Output image info from op is correct
    """
    ms.set_context(device_target="Ascend")

    img = cv2.imread(input_apple_jpg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    img_transformed = vision.AdjustSharpness(sharpness_factor=2.0).device("Ascend")(img)
    logger.info("Image.type: {}, Image.shape: {}".format(img_transformed.dtype, img_transformed.shape))
    assert img_transformed.shape == (2268, 4032, 3)

@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_adjust_sharpness_dvpp_exception():
    """
    Feature: AdjustSharpness op on Ascend910B
    Description: Test eager support for AdjustSharpness with Dvpp when invalid input
    Expectation: Success
    """
    ms.set_context(device_target="Ascend")
    # the input is list
    img = np.ones([1024], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.AdjustSharpness(sharpness_factor=2.0).device("Ascend")(img)
    assert "the input tensor is not HW, HWC or 1HWC," in str(error_info.value)

    # the input is HW2
    img = np.ones([224, 224, 2], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.AdjustSharpness(sharpness_factor=2.0).device("Ascend")(img)
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(error_info.value)

    # the input is 23HW3
    img = np.ones([2, 3, 224, 224, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.AdjustSharpness(sharpness_factor=2.0).device("Ascend")(img)
    assert "The input tensor is not of shape [H,W], [H,W,C] or [N,H,W,C]." in str(error_info.value)

    # the input is 3HW1
    img = np.ones([3, 224, 224, 1], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.AdjustSharpness(sharpness_factor=2.0).device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    # the input is out of [4, 6] to [8192, 4096]
    img = np.ones([3, 6, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.AdjustSharpness(sharpness_factor=2.0).device("Ascend")(img)
    assert "the input shape should be from [4, 6] to [8192, 4096]" in str(error_info.value)

    # the device(device_target) is invalid
    with pytest.raises(TypeError) as error_info:
        _ = vision.AdjustSharpness(sharpness_factor=2.0).device(123)
    assert "Argument device_target with value 123 is not of type [<class 'str'>]" in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        _ = vision.AdjustSharpness(sharpness_factor=2.0).device("Asscend")
    assert "Input device_target is not within the valid set of ['CPU', 'Ascend']" in str(error_info.value)

@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_convert_color_dvpp():
    """
    Feature: ConvertColor op on Ascend910B
    Description: Test eager support for ConvertColor with Dvpp
    Expectation: Output image info from op is correct
    """
    ms.set_context(device_target="Ascend")

    img = cv2.imread(input_apple_jpg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    img_transformed = vision.ConvertColor(vision.ConvertMode.COLOR_RGB2BGR).device("Ascend")(img)
    logger.info("Image.type: {}, Image.shape: {}".format(img_transformed.dtype, img_transformed.shape))
    check_img = cv2.imread(input_apple_jpg)
    check_img = cv2.cvtColor(check_img, cv2.COLOR_BGR2RGB)
    check_img = vision.ConvertColor(vision.ConvertMode.COLOR_RGB2BGR).device("CPU")(check_img)
    assert (img_transformed == check_img).all()

    img = np.random.randint(0, 255, size=(300, 400, 1), dtype=np.uint8)
    img_copy = copy.copy(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    img_transformed = vision.ConvertColor(vision.ConvertMode.COLOR_GRAY2BGRA).device("Ascend")(img)
    logger.info("Image.type: {}, Image.shape: {}".format(img_transformed.dtype, img_transformed.shape))
    # check the result
    check_img = vision.ConvertColor(vision.ConvertMode.COLOR_GRAY2BGRA).device("CPU")(img_copy)
    assert (img_transformed == check_img).all()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_convert_color_dvpp_exception():
    """
    Feature: ConvertColor op on Ascend910B
    Description: Test eager support for ConvertColor with Dvpp when invalid input
    Expectation: Success
    """
    ms.set_context(device_target="Ascend")
    # the input is list
    img = np.ones([1024], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.ConvertColor(vision.ConvertMode.COLOR_RGB2BGR).device("Ascend")(img)
    assert "the input tensor is not HW, HWC or 1HWC," in str(error_info.value)

    # the input is HW2
    img = np.ones([224, 224, 2], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.ConvertColor(vision.ConvertMode.COLOR_RGB2BGR).device("Ascend")(img)
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(error_info.value)

    # the input is 23HW3
    img = np.ones([2, 3, 224, 224, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.ConvertColor(vision.ConvertMode.COLOR_RGB2BGR).device("Ascend")(img)
    assert "The input tensor is not of shape [H,W], [H,W,C] or [N,H,W,C]." in str(error_info.value)

    # the input is 3HW1
    img = np.ones([3, 224, 224, 1], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.ConvertColor(vision.ConvertMode.COLOR_RGB2BGR).device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    # the input is out of [4, 6] to [8192, 4096]
    img = np.ones([3, 6, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.ConvertColor(vision.ConvertMode.COLOR_RGB2BGR).device("Ascend")(img)
    assert "the input shape should be from [4, 6] to [8192, 4096]" in str(error_info.value)

    # the device(device_target) is invalid
    with pytest.raises(TypeError) as error_info:
        _ = vision.ConvertColor(vision.ConvertMode.COLOR_RGB2BGR).device(123)
    assert "Argument device_target with value 123 is not of type [<class 'str'>]" in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        _ = vision.ConvertColor(vision.ConvertMode.COLOR_RGB2BGR).device("Asscend")
    assert "Input device_target is not within the valid set of ['CPU', 'Ascend']" in str(error_info.value)

@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_erase_dvpp():
    """
    Feature: Erase op on Ascend910B
    Description: Test eager support for Erase with Dvpp
    Expectation: Output image info from op is correct
    """
    ms.set_context(device_target="Ascend")

    img = cv2.imread(input_apple_jpg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    img_transformed = vision.Erase(10, 10, 10, 10, (100, 100, 100)).device("Ascend")(img)
    logger.info("Image.type: {}, Image.shape: {}".format(img_transformed.dtype, img_transformed.shape))
    check_img = cv2.imread(input_apple_jpg)
    check_img = cv2.cvtColor(check_img, cv2.COLOR_BGR2RGB)
    check_img = vision.Erase(10, 10, 10, 10, (100, 100, 100)).device("CPU")(check_img)
    assert (img_transformed == check_img).all()

    img = np.random.randint(0, 255, size=(300, 400, 3), dtype=np.uint8)
    img_copy = copy.copy(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    img_transformed = vision.Erase(10, 10, 10, 10, (100, 100, 100)).device("Ascend")(img)
    logger.info("Image.type: {}, Image.shape: {}".format(img_transformed.dtype, img_transformed.shape))
    # check the result
    check_img = vision.Erase(10, 10, 10, 10, (100, 100, 100)).device("CPU")(img_copy)
    assert (img_transformed == check_img).all()

@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_erase_dvpp_exception():
    """
    Feature: Erase op on Ascend910B
    Description: Test eager support for Erase with Dvpp when invalid input
    Expectation: Success
    """
    ms.set_context(device_target="Ascend")
    # the input is list
    img = np.ones([1024], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Erase(10, 10, 10, 10).device("Ascend")(img)
    assert "the input tensor is not HW, HWC or 1HWC," in str(error_info.value)

    # the input is HW2
    img = np.ones([224, 224, 2], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Erase(10, 10, 10, 10).device("Ascend")(img)
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(error_info.value)

    # the input is 23HW3
    img = np.ones([2, 3, 224, 224, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Erase(10, 10, 10, 10).device("Ascend")(img)
    assert "The input tensor is not of shape [H,W], [H,W,C] or [N,H,W,C]." in str(error_info.value)

    # the input is 3HW1
    img = np.ones([3, 224, 224, 1], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Erase(10, 10, 10, 10).device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    # the input is out of [4, 6] to [8192, 4096]
    img = np.ones([3, 6, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Erase(10, 10, 10, 10).device("Ascend")(img)
    assert "the input shape should be from [4, 6] to [8192, 4096]" in str(error_info.value)

    # When The input data is float32, the range of value should be [0, 1]
    img = np.random.randn(30, 60, 3).astype(np.float32)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Erase(1, 4, 20, 30, (30, 5, 10)).device("Ascend")(img)
    assert "When The input data is float32, the range of value should be [0, 1]" in str(error_info.value)

    # The length of value should be the same as the value of channel
    img = np.random.randn(40, 40, 1).astype(np.float32)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Erase(1, 4, 20, 30, (1., 1., 1.)).device("Ascend")(img)
    assert "The length of value should be the same as the value of channel" in str(error_info.value)


    # the device(device_target) is invalid
    with pytest.raises(TypeError) as error_info:
        _ = vision.Erase(10, 10, 10, 10).device(123)
    assert "Argument device_target with value 123 is not of type [<class 'str'>]" in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        _ = vision.Erase(10, 10, 10, 10).device("Asscend")
    assert "Input device_target is not within the valid set of ['CPU', 'Ascend']" in str(error_info.value)

@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_rotate_dvpp():
    """
    Feature: Rotate op on Ascend910B
    Description: Test eager support for Rotate with Dvpp
    Expectation: Output image info from op is correct
    """
    ms.set_context(device_target="Ascend")

    img = cv2.imread(input_apple_jpg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    img_transformed = vision.Rotate(degrees=90.0, resample=Inter.NEAREST, expand=False).device("Ascend")(img)
    logger.info("dvpp out Image.type: {}, Image.shape: {}".format(img_transformed.dtype, img_transformed.shape))
    assert img_transformed.shape == (2268, 4032, 3)

@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_rotate_dvpp_exception():
    """
    Feature: Rotate op on Ascend910B
    Description: Test eager support for Rotate with Dvpp when invalid input
    Expectation: Success
    """
    ms.set_context(device_target="Ascend")
    # the input is list
    img = np.ones([1024], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Rotate(degrees=45).device("Ascend")(img)
    assert "the input tensor is not HW, HWC or 1HWC," in str(error_info.value)

    # the input is HW2
    img = np.ones([224, 224, 2], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Rotate(degrees=45).device("Ascend")(img)
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(error_info.value)

    # the input is 23HW3
    img = np.ones([2, 3, 224, 224, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Rotate(degrees=45).device("Ascend")(img)
    assert "The input tensor is not of shape [H,W], [H,W,C] or [N,H,W,C]." in str(error_info.value)

    # the input is 3HW1
    img = np.ones([3, 224, 224, 1], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Rotate(degrees=45).device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    # the input is out of [4, 6] to [32768, 32768]
    img = np.ones([3, 6, 3], dtype=np.uint8)
    #img = vision.Rotate(degrees=45).device("Ascend")(img)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Rotate(degrees=45).device("Ascend")(img)
    assert "the input shape should be from [4, 6] to [32768, 32768]" in str(error_info.value)

    # the device(device_target) is invalid
    with pytest.raises(TypeError) as error_info:
        _ = vision.Rotate(degrees=45).device(123)
    assert "Argument device_target with value 123 is not of type [<class 'str'>]" in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        _ = vision.Rotate(degrees=45).device("Asscend")
    assert "Input device_target is not within the valid set of ['CPU', 'Ascend']" in str(error_info.value)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_posterize_dvpp():
    """
    Feature: Posterize op on Ascend910B
    Description: Test eager support for Posterize with Dvpp
    Expectation: Output image info from op is correct
    """
    ms.set_context(device_target="Ascend")

    img = cv2.imread(input_apple_jpg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img_transformed = vision.Posterize(4).device("Ascend")(img)
    logger.info("Image.type: {}, Image.shape: {}".format(img_transformed.dtype, img_transformed.shape))

    assert img_transformed.shape == (2268, 4032, 3)

    img_transformed2 = vision.Posterize(4).device("CPU")(img)
    logger.info("Image2.type: {}, Image2.shape: {}".format(img_transformed2.dtype, img_transformed2.shape))

    assert img_transformed2.shape == (2268, 4032, 3)

    assert (img_transformed == img_transformed2).all()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_posterize_dvpp_exception():
    """
    Feature: Posterize op on Ascend910B
    Description: Test eager support for Posterize with Dvpp when invalid input
    Expectation: Success
    """
    ms.set_context(device_target="Ascend")
    # the input is list
    img = np.ones([1024], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Posterize(1).device("Ascend")(img)
        assert "the input tensor is not HW, HWC or 1HWC," in str(error_info.value)

    # the input is HW2
    img = np.ones([224, 224, 2], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Posterize(1).device("Ascend")(img)
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(error_info.value)

    # the input is 3HW1
    img = np.ones([3, 224, 224, 1], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Posterize(1).device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    # the input is 23HW3
    img = np.ones([2, 3, 224, 224, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Posterize(1).device("Ascend")(img)
    assert "The input tensor is not of shape [H,W], [H,W,C] or [N,H,W,C]." in str(error_info.value)

    # the input is out of [4, 6] to [8192, 4096]
    img = np.ones([3, 6, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Posterize(1).device("Ascend")(img)
    assert "the input shape should be from [4, 6] to [8192, 4096]" in str(error_info.value)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_solarize_dvpp():
    """
    Feature: Solarize op on Ascend910B
    Description: Test eager support for Solarize with Dvpp
    Expectation: Output image info from op is correct
    """
    ms.set_context(device_target="Ascend")

    img = cv2.imread(input_apple_jpg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img_transformed = vision.Solarize(threshold=10).device("Ascend")(img)
    logger.info("Image.type: {}, Image.shape: {}".format(img_transformed.dtype, img_transformed.shape))

    assert img_transformed.shape == (2268, 4032, 3)

    img_transformed2 = vision.Solarize(threshold=10).device("CPU")(img)
    logger.info("Image2.type: {}, Image2.shape: {}".format(img_transformed2.dtype, img_transformed2.shape))

    assert img_transformed2.shape == (2268, 4032, 3)

    assert (img_transformed == img_transformed2).all()



@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_eager_solarize_dvpp_exception():
    """
    Feature: Solarize op on Ascend910B
    Description: Test eager support for Solarize with Dvpp when invalid input
    Expectation: Success
    """
    ms.set_context(device_target="Ascend")
    # the input is list
    img = np.ones([1024], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Solarize(threshold=10).device("Ascend")(img)
        assert "the input tensor is not HW, HWC or 1HWC," in str(error_info.value)

    # the input is HW2
    img = np.ones([224, 224, 2], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Solarize(threshold=10).device("Ascend")(img)
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(error_info.value)

    # the input is 3HW1
    img = np.ones([3, 224, 224, 1], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Solarize(threshold=10).device("Ascend")(img)
    assert "The input tensor NHWC should be 1HWC or HWC." in str(error_info.value)

    # the input is 23HW3
    img = np.ones([2, 3, 224, 224, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Solarize(threshold=10).device("Ascend")(img)
    assert "The input tensor is not of shape [H,W], [H,W,C] or [N,H,W,C]." in str(error_info.value)

    # the input is out of [4, 6] to [8192, 4096]
    img = np.ones([3, 6, 3], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Solarize(threshold=10).device("Ascend")(img)
    assert "the input shape should be from [4, 6] to [8192, 4096]" in str(error_info.value)


def test_resize_performance():
    """
    Feature: Resize
    Description: Test dvpp Resize performance in eager mode after optimize ndarray to cde.Tensor without memcpy
    Expectation: SUCCESS
    """

    img_bytes = np.fromfile(input_apple_jpg, dtype=np.uint8)
    img_decode = vision.Decode()(img_bytes)
    _ = vision.Resize(224).device("Ascend")(img_decode)

    s = time.time()
    for _ in range(1000):
        _ = vision.Resize(224).device("Ascend")(img_decode)
    assert (time.time() - s) < 5.0  # Probably around 4.43 seconds


if __name__ == '__main__':
    test_eager_resize_dvpp()
    test_resize_performance()
    test_eager_resize_dvpp_exception()
    test_eager_resize_dvpp_exception_with_910A()
    test_eager_decode_dvpp()
    test_eager_decode_dvpp_exception()
    test_eager_normalize_dvpp()
    test_eager_normalize_dvpp_exception()
    test_eager_multi_dvpp_op_dvpp_cpu_dvpp()
    test_eager_multi_dvpp_op_dvpp_dvpp_cpu()
    test_eager_multi_dvpp_op_cpu_dvpp_dvpp()
    test_eager_compose_dvpp_ops()
    test_eager_horizontal_flip_dvpp()
    test_eager_horizontal_flip_dvpp_exception()
    test_eager_vertical_flip_dvpp()
    test_eager_vertical_flip_dvpp_exception()
    test_eager_resize_crop_dvpp()
    test_eager_resize_crop_dvpp_exception()
    test_eager_perspective_dvpp()
    test_eager_perspective_dvpp_exception()
    test_eager_equalize_dvpp()
    test_eager_equalize_dvpp_exception()
    test_eager_auto_contrast_dvpp()
    test_eager_auto_contrast_dvpp_exception()
    test_eager_adjust_sharpness_dvpp()
    test_eager_adjust_sharpness_dvpp_exception()
    test_eager_convert_color_dvpp()
    test_eager_convert_color_dvpp_exception()
    test_eager_erase_dvpp()
    test_eager_erase_dvpp_exception()
    test_eager_rotate_dvpp()
    test_eager_rotate_dvpp_exception()
    test_eager_posterize_dvpp()
    test_eager_posterize_dvpp_exception()
    test_eager_solarize_dvpp()
    test_eager_solarize_dvpp_exception()
