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
import cv2
import numpy as np
from PIL import Image
import pytest

from mindspore import log as logger
from mindspore import Tensor
from mindspore.dataset import transforms
from mindspore.dataset import vision


def test_eager_decode_c():
    """
    Feature: Decode op
    Description: Test eager support for Decode Cpp implementation
    Expectation: Output image size from op is correct
    """
    img = np.fromfile("../data/dataset/apple.jpg", dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img = vision.Decode()(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    assert img.shape == (2268, 4032, 3)

    with open("../data/dataset/apple.jpg", "rb") as fp:
        img2 = fp.read()

    img2 = vision.Decode()(img2)
    logger.info("Image.type: {}, Image.shape: {}".format(
        type(img2), img2.shape))
    assert img2.shape == (2268, 4032, 3)


def test_eager_decode_py():
    """
    Feature: Decode op
    Description: Test eager support for Decode Python implementation
    Expectation: Output image size from op is correct
    """
    img = np.fromfile("../data/dataset/apple.jpg", dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))

    img = vision.Decode(to_pil=True)(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))
    assert img.size == (4032, 2268)

    with open("../data/dataset/apple.jpg", "rb") as fp:
        img2 = fp.read()

    img2 = vision.Decode(to_pil=True)(img2)
    logger.info("Image.type: {}, Image.shape: {}".format(
        type(img2), img2.size))
    assert img2.size == (4032, 2268)


def test_eager_resize_c():
    """
    Feature: Resize op
    Description: Test eager support for Resize C++ implementation
    Expectation: Output image size from op is correct
    """
    img = cv2.imread("../data/dataset/apple.jpg")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img = vision.Resize(size=(64, 32))(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    assert img.shape == (64, 32, 3)


def test_eager_resize_py():
    """
    Feature: Resize op
    Description: Test eager support for Resize Python implementation
    Expectation: Output image info from op is correct
    """
    img = Image.open("../data/dataset/apple.jpg").convert("RGB")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))

    img = vision.Resize(size=(96, 64))(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))

    assert img.size == (64, 96)


def test_eager_rescale():
    """
    Feature: Rescale op
    Description: Test eager support for Rescale op
    Expectation: Output image info from op is correct
    """
    img = cv2.imread("../data/dataset/apple.jpg")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    pixel = img[0][0][0]

    rescale_factor = 0.5
    img = vision.Rescale(rescale=rescale_factor, shift=0)(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    pixel_rescaled = img[0][0][0]

    assert pixel * rescale_factor == pixel_rescaled


def test_eager_normalize_hwc():
    """
    Feature: Normalize op
    Description: Test eager support for Normalize with HWC shape
    Expectation: Output image info from op is correct
    """
    img = Image.open("../data/dataset/apple.jpg").convert("RGB")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))
    pixel = img.getpixel((0, 0))[0]

    mean_vec = [100, 100, 100]
    std_vec = [2, 2, 2]
    img = vision.Normalize(mean=mean_vec, std=std_vec, is_hwc=True)(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    pixel_normalized = img[0][0][0]

    assert (pixel - mean_vec[0]) / std_vec[0] == pixel_normalized


def test_eager_normalize_chw():
    """
    Feature: Normalize op
    Description: Test eager support for Normalize with CHW shape
    Expectation: Output image info from op is correct
    """
    img = Image.open("../data/dataset/apple.jpg").convert("RGB")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))
    pixel = img.getpixel((0, 0))[0]

    img = vision.ToTensor()(img)

    mean_vec = [.100, .100, .100]
    std_vec = [.2, .2, .2]
    img = vision.Normalize(mean=mean_vec, std=std_vec, is_hwc=False)(img)
    pixel_normalized = img[0][0][0]

    assert (pixel / 255 - mean_vec[0]) / \
           std_vec[0] == pytest.approx(pixel_normalized, 0.0001)


def test_eager_resize_totensor_normalize_py():
    """
    Feature: Eager Support
    Description: Test eager support for this sequence of Python ops: Resize, ToTensor and Normalize
    Expectation: Output image info from op is correct
    """
    img = Image.open("../data/dataset/apple.jpg").convert("RGB")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))

    img = vision.Resize(size=(96, 64))(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))

    assert img.size == (64, 96)

    pixel = img.getpixel((0, 0))[0]

    img = vision.ToTensor()(img)

    mean_vec = [.100, .100, .100]
    std_vec = [.2, .2, .2]
    img = vision.Normalize(mean=mean_vec, std=std_vec, is_hwc=False)(img)
    pixel_normalized = img[0][0][0]

    assert img.size == 64 * 96 * 3

    assert (pixel / 255 - mean_vec[0]) / std_vec[0] == pytest.approx(pixel_normalized, 0.0001)


def test_eager_compose_py():
    """
    Feature: Eager Support
    Description: Test eager support for this sequence of Python ops: Resize, Compose with ToTensor and Normalize
    Expectation: Output image info from op is correct
    """
    img = Image.open("../data/dataset/apple.jpg").convert("RGB")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))

    img = vision.Resize(size=(96, 64))(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))

    assert img.size == (64, 96)

    pixel = img.getpixel((0, 0))[0]

    mean_vec = [.100, .100, .100]
    std_vec = [.2, .2, .2]

    transform = transforms.Compose([
        vision.ToTensor(),
        vision.Normalize(mean=mean_vec, std=std_vec, is_hwc=False)])

    # Convert to NumPy array
    img = np.array(img)
    output_size = 64 * 96 * 3
    assert img.size == output_size

    # Use Compose to apply transformation with ToTensor and Normalize
    # Note: Output of Compose is a NumPy array
    img = transform(img)
    assert img.size == output_size
    assert isinstance(img, np.ndarray)

    pixel_normalized = img[0][0][0]

    assert (pixel / 255 - mean_vec[0]) / std_vec[0] == pytest.approx(pixel_normalized, 0.0001)


def test_eager_hwc2chw():
    """
    Feature: HWC2CHW op
    Description: Test eager support for HWC2CHW op
    Expectation: Output image size from op is correct
    """
    img = cv2.imread("../data/dataset/apple.jpg")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    channel = img.shape

    img = vision.HWC2CHW()(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    channel_swapped = img.shape

    assert channel == (channel_swapped[1],
                       channel_swapped[2], channel_swapped[0])


def test_eager_pad_c():
    """
    Feature: Pad op
    Description: Test eager support for Pad Cpp implementation
    Expectation: Output image size info from op is correct
    """
    img = cv2.imread("../data/dataset/apple.jpg")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img = vision.Resize(size=(32, 32))(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    shape_org = img.shape

    pad = 4
    img = vision.Pad(padding=pad)(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    shape_padded = img.shape

    assert shape_padded == (
        shape_org[0] + 2 * pad, shape_org[1] + 2 * pad, shape_org[2])


def test_eager_pad_py():
    """
    Feature: Pad op
    Description: Test eager support for Pad Python implementation
    Expectation: Output image size info from op is correct
    """
    img = Image.open("../data/dataset/apple.jpg").convert("RGB")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))

    img = vision.Resize(size=(32, 32))(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))
    size = img.size

    pad = 4
    img = vision.Pad(padding=pad)(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))
    size_padded = img.size

    assert size_padded == (size[0] + 2 * pad, size[1] + 2 * pad)


def test_eager_cutout_hwc_pil():
    """
    Feature: CutOut op
    Description: Test eager support for CutOut with HWC shape and PIL input
    Expectation: Output image size info from op is correct
    """
    img = Image.open("../data/dataset/apple.jpg").convert("RGB")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))

    img = vision.Resize(size=(32, 32))(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))
    size = img.size

    img = vision.CutOut(2, 4)(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    size_cutout = img.shape

    assert (size_cutout[0], size_cutout[1]) == size


def test_eager_cutout_chw_pil():
    """
    Feature: CutOut op
    Description: Test eager support for CutOut with CHW shape and PIL input
    Expectation: Receive non-None output image from op
    """
    img = Image.open("../data/dataset/apple.jpg").convert("RGB")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))

    img = vision.Resize(size=(32, 32))(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))

    img = vision.ToTensor()(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))

    img = vision.CutOut(2, 4, is_hwc=False)(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))

    assert img is not None


def test_eager_cutout_hwc_cv():
    """
    Feature: CutOut op
    Description: Test eager support for CutOut with HWC shape and CV input
    Expectation: Output image size info from op is correct
    """
    img = cv2.imread("../data/dataset/apple.jpg")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))

    img = vision.Resize(size=(32, 32))(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))
    size = img.size

    img = vision.CutOut(2, 4)(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))
    size_cutout = img.size

    assert size_cutout == size


def test_eager_exceptions_decode():
    """
    Feature: Decode op
    Description: Exception eager support test for Decode
    Expectation: Error input image is detected
    """
    with pytest.raises(TypeError) as error_info:
        img = "../data/dataset/apple.jpg"
        _ = vision.Decode()(img)
    assert "The type of the encoded image should be <class 'numpy.ndarray'>" in str(error_info.value)

    with pytest.raises(TypeError) as error_info:
        img = np.array(["a", "b", "c"])
        _ = vision.Decode()(img)
    assert "The data type of the encoded image can not be <class 'numpy.str_'>" in str(error_info.value)


def test_eager_exceptions_resize():
    """
    Feature: Resize op
    Description: Exception eager support test for Resize Python implementation
    Expectation: Error input image is detected
    """
    try:
        img = cv2.imread("../data/dataset/apple.jpg")
        _ = vision.Resize(size=(-32, 32))(img)
        assert False
    except ValueError as e:
        assert "not within the required interval" in str(e)


def test_eager_exceptions_normalize():
    """
    Feature: Normalize op
    Description: Exception eager support test for Normalize Python implementation
    Expectation: Error input image is detected
    """
    try:
        img = Image.open("../data/dataset/apple.jpg").convert("RGB")
        mean_vec = [.100, .100, .100]
        std_vec = [.2, .2, .2]
        _ = vision.Normalize(mean=mean_vec, std=std_vec, is_hwc=False)(img)
        assert False
    except RuntimeError as e:
        assert "Normalize: number of channels does not match the size of mean and std vectors" in str(
            e)


def test_eager_exceptions_pad():
    """
    Feature: Pad
    Description: Test Pad with invalid input of string
    Expectation: Raise TypeError
    """
    try:
        img = "../data/dataset/apple.jpg"
        _ = vision.Pad(padding=4)(img)
    except TypeError as e:
        assert "Input should be NumPy or PIL image, got <class 'str'>." in str(e)


def test_eager_invalid_image_randomadjustsharpness():
    """
    Feature: RandomAdjustSharpness op
    Description: Exception eager support test for RandomAdjustSharpness op with invalid image input type
    Expectation: Error input image is detected
    """

    def test_config(my_input, error_type, error_msg):
        with pytest.raises(error_type) as error_info:
            _ = vision.RandomAdjustSharpness(degree=0.5, prob=1)(my_input)
        assert error_msg in str(error_info.value)

    my_input = np.random.rand(128, 128, 3)

    test_config((10,), TypeError, "Input should be NumPy or PIL image, got <class 'tuple'>.")
    test_config(10, TypeError, "Input should be NumPy or PIL image, got <class 'int'>.")
    test_config(Tensor(my_input), TypeError,
                "Input should be NumPy or PIL image, got <class 'mindspore.common.tensor.Tensor'>.")


def test_eager_invalid_image_hwc2chw():
    """
    Feature: HWC2CHW op
    Description: Exception eager support test for HWC2CHW op with invalid image input type
    Expectation: Error input image is detected
    """

    def test_config(my_input, error_type, error_msg):
        with pytest.raises(error_type) as error_info:
            _ = vision.HWC2CHW()(my_input)
        assert error_msg in str(error_info.value)

    my_input = np.random.randn(64, 32, 3).astype(np.int32).tolist()
    test_config(my_input, TypeError, "Input should be NumPy or PIL image, got <class 'list'>.")
    test_config(Tensor(my_input), TypeError,
                "Input should be NumPy or PIL image, got <class 'mindspore.common.tensor.Tensor'>.")


def test_eager_invalid_image_invert():
    """
    Feature: Invert op
    Description: Exception eager support test for Invert op with invalid image type
    Expectation: Error input image is detected
    """

    def test_config(my_input, error_type, error_msg):
        with pytest.raises(error_type) as error_info:
            _ = vision.Invert()(my_input)
        assert error_msg in str(error_info.value)

    test_config((10,), TypeError, "Input should be NumPy or PIL image, got <class 'tuple'>.")
    test_config(10, TypeError, "Input should be NumPy or PIL image, got <class 'int'>.")


def test_eager_invalid_image_pad():
    """
    Feature: Pad op
    Description: Exception eager support test for Pad op with invalid image input type
    Expectation: Error input image is detected
    """

    def test_config(my_input, error_type, error_msg):
        with pytest.raises(error_type) as error_info:
            _ = vision.Pad(padding=10)(my_input)
        assert error_msg in str(error_info.value)

    my_input = np.random.randn(64, 32, 3).astype(np.int32).tolist()
    test_config(my_input, TypeError, "Input should be NumPy or PIL image, got <class 'list'>.")
    test_config(Tensor(my_input), TypeError,
                "Input should be NumPy or PIL image, got <class 'mindspore.common.tensor.Tensor'>.")


def test_eager_invalid_image_randomcrop():
    """
    Feature: RandomCrop op
    Description: Exception eager support test for RandomCrop op with invalid image input type
    Expectation: Error input image is detected
    """

    def test_config(my_input, error_type, error_msg):
        with pytest.raises(error_type) as error_info:
            _ = vision.RandomCrop(size=200)(my_input)
        assert error_msg in str(error_info.value)

    my_input = np.random.randint(0, 255, (987, 654, 3)).astype(np.uint8).tolist()
    test_config(my_input, TypeError, "Input should be NumPy or PIL image, got <class 'list'>.")


def test_eager_invalid_image_randomhorizontalflip():
    """
    Feature: RandomHorizontalFlip op
    Description: Exception eager support test for RandomHorizontalFlip op with invalid image input type
    Expectation: Error input image is detected
    """

    def test_config(my_input, error_type, error_msg):
        with pytest.raises(error_type) as error_info:
            _ = vision.RandomHorizontalFlip(prob=1)(my_input)
        assert error_msg in str(error_info.value)

    img = cv2.imread("../data/dataset/apple.jpg")
    my_input = img.tolist()
    test_config(my_input, TypeError, "Input should be NumPy or PIL image, got <class 'list'>.")


def test_eager_invalid_image_randomsolarize():
    """
    Feature: RandomSolarize op
    Description: Exception eager support test for RandomSolarize op with invalid image input type
    Expectation: Error input image is detected
    """

    def test_config(my_input, error_type, error_msg):
        with pytest.raises(error_type) as error_info:
            _ = vision.RandomSolarize(threshold=(0, 120))(my_input)
        assert error_msg in str(error_info.value)

    my_input = np.random.randint(0, 255, (500, 600, 3)).astype(np.uint8).tolist()
    test_config(my_input, TypeError, "Input should be NumPy or PIL image, got <class 'list'>.")


def test_eager_invalid_image_cutout():
    """
    Feature: CutOut op
    Description: Exception eager support test for CutOut op with invalid image input type
    Expectation: Error input image is detected
    """

    def test_config(my_input, error_type, error_msg):
        with pytest.raises(error_type) as error_info:
            _ = vision.CutOut(length=120, num_patches=1)(my_input)
        assert error_msg in str(error_info.value)

    my_input = np.random.randn(60, 50)
    test_config(my_input, RuntimeError, "CutOut: shape is invalid.")

    test_config(1, TypeError, "Input should be NumPy or PIL image, got <class 'int'>.")
    test_config(1.0, TypeError, "Input should be NumPy or PIL image, got <class 'float'>.")
    test_config((10, 20), TypeError, "Input should be NumPy or PIL image, got <class 'tuple'>.")
    test_config([10, 20, 30], TypeError, "Input should be NumPy or PIL image, got <class 'list'>.")


def test_eager_invalid_image_randomcolor():
    """
    Feature: RandomColor op
    Description: Exception eager support test for RandomColor op with invalid image input type
    Expectation: Error input image is detected
    """

    def test_config(my_input, error_type, error_msg):
        with pytest.raises(error_type) as error_info:
            _ = vision.RandomColor(degrees=(0.2, 0.3))(my_input)
        assert error_msg in str(error_info.value)

    my_input = np.random.randn(1280, 1280, 3)

    test_config(None, TypeError, "Input should be NumPy or PIL image, got <class 'NoneType'>.")
    test_config(1, TypeError, "Input should be NumPy or PIL image, got <class 'int'>.")
    test_config(1.0, TypeError, "Input should be NumPy or PIL image, got <class 'float'>.")
    test_config((10, 20), TypeError, "Input should be NumPy or PIL image, got <class 'tuple'>.")
    test_config([10, 20, 30], TypeError, "Input should be NumPy or PIL image, got <class 'list'>.")
    test_config(Tensor(my_input), TypeError,
                "Input should be NumPy or PIL image, got <class 'mindspore.common.tensor.Tensor'>.")


def test_eager_invalid_image_randomsharpness():
    """
    Feature: RandomSharpness op
    Description: Exception eager support test for RandomSharpness op with invalid image input type
    Expectation: Error input image is detected
    """

    def test_config(my_input, error_type, error_msg):
        with pytest.raises(error_type) as error_info:
            _ = vision.RandomSharpness(degrees=(0.2, 0.3))(my_input)
        assert error_msg in str(error_info.value)

    my_input = np.random.randn(1280, 1280, 3)

    test_config(None, TypeError, "Input should be NumPy or PIL image, got <class 'NoneType'>.")
    test_config(1, TypeError, "Input should be NumPy or PIL image, got <class 'int'>.")
    test_config(1.0, TypeError, "Input should be NumPy or PIL image, got <class 'float'>.")
    test_config((10, 20), TypeError, "Input should be NumPy or PIL image, got <class 'tuple'>.")
    test_config([10, 20, 30], TypeError, "Input should be NumPy or PIL image, got <class 'list'>.")
    test_config(Tensor(my_input), TypeError,
                "Input should be NumPy or PIL image, got <class 'mindspore.common.tensor.Tensor'>.")


if __name__ == '__main__':
    test_eager_decode_c()
    test_eager_decode_py()
    test_eager_resize_c()
    test_eager_resize_py()
    test_eager_rescale()
    test_eager_normalize_hwc()
    test_eager_normalize_chw()
    test_eager_resize_totensor_normalize_py()
    test_eager_compose_py()
    test_eager_hwc2chw()
    test_eager_pad_c()
    test_eager_pad_py()
    test_eager_cutout_hwc_pil()
    test_eager_cutout_chw_pil()
    test_eager_cutout_hwc_cv()
    test_eager_exceptions_decode()
    test_eager_exceptions_resize()
    test_eager_exceptions_normalize()
    test_eager_exceptions_pad()
    test_eager_invalid_image_randomadjustsharpness()
    test_eager_invalid_image_hwc2chw()
    test_eager_invalid_image_invert()
    test_eager_invalid_image_pad()
    test_eager_invalid_image_randomcrop()
    test_eager_invalid_image_randomhorizontalflip()
    test_eager_invalid_image_randomsolarize()
    test_eager_invalid_image_cutout()
    test_eager_invalid_image_randomcolor()
    test_eager_invalid_image_randomsharpness()
