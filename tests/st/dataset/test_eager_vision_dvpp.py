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
import cv2
import numpy as np
import pytest

from mindspore import log as logger
import mindspore as ms
import mindspore.dataset.vision as vision
from utils import ascend910b

# pylint: disable=W0212
# W0212: protected-access


input_apple_jpg = "/home/workspace/mindspore_dataset/910B_dvpp/apple.jpg"
result_data_dir = "/home/workspace/mindspore_dataset/910B_dvpp/testAscend910BDvpp"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@ascend910b
def test_eager_decode_dvpp():
    """
    Feature: Decode op when Ascend910B
    Description: Test eager support for Decode with Dvpp
    Expectation: Output image info from op is correct
    """
    os.environ['MS_ENABLE_REF_MODE'] = "1"
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

    os.environ['MS_ENABLE_REF_MODE'] = "0"


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@ascend910b
def test_eager_decode_dvpp_exception():
    """
    Feature: Decode op when Ascend910B with exception
    Description: Test eager support for Decode with Dvpp
    Expectation: Output image info from op is correct
    """
    os.environ['MS_ENABLE_REF_MODE'] = "1"
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

    os.environ['MS_ENABLE_REF_MODE'] = "0"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@ascend910b
def test_eager_resize_dvpp():
    """
    Feature: Resize op when Ascend910B
    Description: Test eager support for Resize with Dvpp
    Expectation: Output image info from op is correct
    """
    os.environ['MS_ENABLE_REF_MODE'] = "1"
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

    os.environ['MS_ENABLE_REF_MODE'] = "0"


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@ascend910b
def test_eager_resize_dvpp_exception():
    """
    Feature: Resize op when Ascend910B
    Description: Test eager support for Resize with Dvpp when invalid input
    Expectation: Success
    """
    os.environ['MS_ENABLE_REF_MODE'] = "1"
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
    assert "The channel of the input tensor of shape [H,W,C] is not 1 or 3" in str(error_info.value)

    # the input is HW4
    img = np.ones([224, 224, 4], dtype=np.uint8)
    with pytest.raises(RuntimeError) as error_info:
        img = vision.Resize(size=(64, 32)).device("Ascend")(img)
    assert "The channel of the input tensor of shape [H,W,C] is not 1 or 3" in str(error_info.value)

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
    assert "The current InterpolationMode is not supported by DVPP. It is " in str(error_info.value)

    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Resize(size=(64, 32), interpolation=vision.Inter.PILCUBIC).device("Ascend")(img)
    assert "The current InterpolationMode is not supported by DVPP. It is " in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        _ = vision.Resize(size=(64, 32), interpolation=vision.Inter.ANTIALIAS).device("Ascend")(img)
    assert "The current InterpolationMode is not supported by DVPP. It is " in str(error_info.value)

    # the device(device_target) is invalid
    with pytest.raises(TypeError) as error_info:
        _ = vision.Resize(size=(64, 32)).device(12)
    assert "Argument device_target with value 12 is not of type [<class 'str'>]" in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        _ = vision.Resize(size=(64, 32)).device("Asscend")
    assert "Input device_target is not within the valid set of ['CPU', 'Ascend']" in str(error_info.value)

    os.environ['MS_ENABLE_REF_MODE'] = "0"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@ascend910b
def test_eager_normalize_dvpp():
    """
    Feature: Normalize op when Ascend910B
    Description: Test eager support for Normalize with Dvpp
    Expectation: Output image info from op is correct
    """
    os.environ['MS_ENABLE_REF_MODE'] = "1"
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

    os.environ['MS_ENABLE_REF_MODE'] = "0"


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@ascend910b
def test_eager_normalize_dvpp_exception():
    """
    Feature: Normalize op when Ascend910B with exception
    Description: Test eager support for Normalize with Dvpp
    Expectation: Output image info from op is correct
    """
    os.environ['MS_ENABLE_REF_MODE'] = "1"
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
    assert "The channel of the input tensor of shape [H,W,C] is not 1 or 3" in str(error_info.value)

    # HW2
    img = np.ones([224, 224, 2], dtype=np.uint8) * 221
    with pytest.raises(RuntimeError) as error_info:
        _ = vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img)
    assert "The channel of the input tensor of shape [H,W,C] is not 1 or 3" in str(error_info.value)

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

    os.environ['MS_ENABLE_REF_MODE'] = "0"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@ascend910b
def test_eager_multi_dvpp_op_dvpp_cpu_dvpp():
    """
    Feature: Multi ops when Ascend910B with global executor
    Description: Test eager support for multi op with Dvpp
    Expectation: Output image info from op is correct
    """
    os.environ['MS_ENABLE_REF_MODE'] = "1"
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

    os.environ['MS_ENABLE_REF_MODE'] = "0"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@ascend910b
def test_eager_multi_dvpp_op_dvpp_dvpp_cpu():
    """
    Feature: Multi ops when Ascend910B with global executor
    Description: Test eager support for multi op with Dvpp
    Expectation: Output image info from op is correct
    """
    os.environ['MS_ENABLE_REF_MODE'] = "1"
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

    os.environ['MS_ENABLE_REF_MODE'] = "0"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@ascend910b
def test_eager_multi_dvpp_op_cpu_dvpp_dvpp():
    """
    Feature: Multi ops when Ascend910B with global executor
    Description: Test eager support for multi op with Dvpp
    Expectation: Output image info from op is correct
    """
    os.environ['MS_ENABLE_REF_MODE'] = "1"
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

    os.environ['MS_ENABLE_REF_MODE'] = "0"


if __name__ == '__main__':
    test_eager_resize_dvpp()
    test_eager_resize_dvpp_exception()
    test_eager_decode_dvpp()
    test_eager_decode_dvpp_exception()
    test_eager_normalize_dvpp()
    test_eager_normalize_dvpp_exception()
    test_eager_multi_dvpp_op_dvpp_cpu_dvpp()
    test_eager_multi_dvpp_op_dvpp_dvpp_cpu()
    test_eager_multi_dvpp_op_cpu_dvpp_dvpp()
