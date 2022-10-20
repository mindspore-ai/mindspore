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
import pytest
from PIL import Image
import mindspore.dataset.transforms.py_transforms as PT
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.vision.py_transforms as PY
from mindspore import log as logger


def test_eager_decode_c():
    """
    Feature: Decode op
    Description: Test eager support for Decode C++ op
    Expectation: Output image size from op is correct
    """
    img = np.fromfile("../data/dataset/apple.jpg", dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img = C.Decode()(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    assert img.shape == (2268, 4032, 3)

    fp = open("../data/dataset/apple.jpg", "rb")
    img2 = fp.read()

    img2 = C.Decode()(img2)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img2), img2.shape))
    assert img2.shape == (2268, 4032, 3)


def test_eager_decode_py():
    """
    Feature: Decode op
    Description: Test eager support for Decode Python op
    Expectation: Output image size from op is correct
    """
    img = np.fromfile("../data/dataset/apple.jpg", dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))

    img = PY.Decode()(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))
    assert img.size == (4032, 2268)

    fp = open("../data/dataset/apple.jpg", "rb")
    img2 = fp.read()

    img2 = PY.Decode()(img2)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img2), img2.size))
    assert img2.size == (4032, 2268)


def test_eager_resize_c():
    """
    Feature: Resize op
    Description: Test eager support for Resize C++ op
    Expectation: Output image info from op is correct
    """
    img = cv2.imread("../data/dataset/apple.jpg")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img = C.Resize(size=(64, 32))(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    assert img.shape == (64, 32, 3)


def test_eager_resize_py():
    """
    Feature: Resize op
    Description: Test eager support for Resize Python op
    Expectation: Output image info from op is correct
    """
    img = Image.open("../data/dataset/apple.jpg").convert("RGB")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))

    img = PY.Resize(size=(96, 64))(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))

    assert img.size == (64, 96)


def test_eager_rescale():
    """
    Feature: Rescale op
    Description: Test eager support for Rescale C++ op
    Expectation: Output image info from op is correct
    """
    img = cv2.imread("../data/dataset/apple.jpg")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    pixel = img[0][0][0]

    rescale_factor = 0.5
    img = C.Rescale(rescale=rescale_factor, shift=0)(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    pixel_rescaled = img[0][0][0]

    assert pixel * rescale_factor == pixel_rescaled


def test_eager_normalize_c():
    """
    Feature: Normalize op
    Description: Test eager support for Normalize C++ op
    Expectation: Output image info from op is correct
    """
    img = Image.open("../data/dataset/apple.jpg").convert("RGB")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))
    pixel = img.getpixel((0, 0))[0]

    mean_vec = [100, 100, 100]
    std_vec = [2, 2, 2]
    img = C.Normalize(mean=mean_vec, std=std_vec)(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    pixel_normalized = img[0][0][0]

    assert (pixel - mean_vec[0]) / std_vec[0] == pixel_normalized


def test_eager_normalize_py():
    """
    Feature: Normalize op
    Description: Test eager support for Normalize Python op
    Expectation: Output image info from op is correct
    """
    img = Image.open("../data/dataset/apple.jpg").convert("RGB")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))
    pixel = img.getpixel((0, 0))[0]

    img = PY.ToTensor()(img)

    mean_vec = [.100, .100, .100]
    std_vec = [.2, .2, .2]
    img = PY.Normalize(mean=mean_vec, std=std_vec)(img)
    pixel_normalized = img[0][0][0]

    assert (pixel / 255 - mean_vec[0]) / std_vec[0] == pytest.approx(pixel_normalized, 0.0001)


def test_eager_resize_totensor_normalize_py():
    """
    Feature: Eager Support
    Description: Test eager support for this sequence of Python ops: Resize, ToTensor and Normalize
    Expectation: Output image info from op is correct
    """
    img = Image.open("../data/dataset/apple.jpg").convert("RGB")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))

    img = PY.Resize(size=(96, 64))(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))

    assert img.size == (64, 96)

    pixel = img.getpixel((0, 0))[0]

    img = PY.ToTensor()(img)

    mean_vec = [.100, .100, .100]
    std_vec = [.2, .2, .2]
    img = PY.Normalize(mean=mean_vec, std=std_vec)(img)
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

    img = PY.Resize(size=(96, 64))(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))

    assert img.size == (64, 96)

    pixel = img.getpixel((0, 0))[0]

    mean_vec = [.100, .100, .100]
    std_vec = [.2, .2, .2]

    transform = PT.Compose([
        PY.ToTensor(),
        PY.Normalize(mean=mean_vec, std=std_vec)])

    # Convert to NumPy array
    img = np.array(img)
    output_size = 64 * 96 * 3
    assert img.size == output_size

    # Use Compose to apply transformation with ToTensor and Normalize
    # Note: Output of Compose is a tuple
    img = transform(img)
    assert isinstance(img, tuple)

    pixel_normalized = img[0][0][0]

    assert (pixel / 255 - mean_vec[0]) / std_vec[0] == pytest.approx(pixel_normalized[0], 0.0001)


def test_eager_hwc2chw():
    """
    Feature: HWC2CHW op
    Description: Test eager support for HWC2CHW C++ op
    Expectation: Output image info from op is correct
    """
    img = cv2.imread("../data/dataset/apple.jpg")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    channel = img.shape

    img = C.HWC2CHW()(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    channel_swaped = img.shape

    assert channel == (channel_swaped[1], channel_swaped[2], channel_swaped[0])


def test_eager_pad_c():
    """
    Feature: Pad op
    Description: Test eager support for Pad C++ op
    Expectation: Output image size info from op is correct
    """
    img = Image.open("../data/dataset/apple.jpg").convert("RGB")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))

    img = C.Resize(size=(32, 32))(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))
    size = img.shape

    pad = 4
    img = C.Pad(padding=pad)(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))
    size_padded = img.shape

    assert size_padded == (size[0] + 2 * pad, size[1] + 2 * pad, size[2])


def test_eager_pad_py():
    """
    Feature: Pad op
    Description: Test eager support for Pad Python op
    Expectation: Output image size info from op is correct
    """
    img = Image.open("../data/dataset/apple.jpg").convert("RGB")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))

    img = PY.Resize(size=(32, 32))(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))
    size = img.size

    pad = 4
    img = PY.Pad(padding=pad)(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))
    size_padded = img.size

    assert size_padded == (size[0] + 2 * pad, size[1] + 2 * pad)


def test_eager_cutout_pil_c():
    """
    Feature: CutOut op
    Description: Test eager support for CutOut C++ op with PIL input
    Expectation: Output image size info from op is correct
    """
    img = Image.open("../data/dataset/apple.jpg").convert("RGB")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))

    img = C.Resize(size=(32, 32))(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))
    size = img.shape

    img = C.CutOut(2, 4)(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))
    size_cutout = img.shape

    assert size_cutout == size


def test_eager_cutout_pil_py():
    """
    Feature: CutOut op
    Description: Test eager support for CutOut Python op with PIL input
    Expectation: Receive non-None output image from op
    """
    img = Image.open("../data/dataset/apple.jpg").convert("RGB")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))

    img = PY.Resize(size=(32, 32))(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))

    img = PY.ToTensor()(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))

    img = PY.Cutout(2, 4)(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))

    assert img is not None


def test_eager_cutout_cv_c():
    """
    Feature: CutOut op
    Description: Test eager support for CutOut C++ op with CV input
    Expectation: Output image size info from op is correct
    """
    img = cv2.imread("../data/dataset/apple.jpg")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))

    img = C.Resize(size=(32, 32))(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))
    size = img.shape

    img = C.CutOut(2, 4)(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))
    size_cutout = img.shape

    assert size_cutout == size


def test_eager_exceptions():
    """
    Feature: Eager Support
    Description: Exception eager support test for various vision C++ ops
    Expectation: Error input image is detected
    """
    try:
        img = "../data/dataset/apple.jpg"
        _ = C.Decode()(img)
        assert False
    except TypeError as e:
        assert "Input should be an encoded image in 1-D NumPy format" in str(e)

    try:
        img = np.array(["a", "b", "c"])
        _ = C.Decode()(img)
        assert False
    except TypeError as e:
        assert "Input should be an encoded image in 1-D NumPy format" in str(e)

    try:
        img = cv2.imread("../data/dataset/apple.jpg")
        _ = C.Resize(size=(-32, 32))(img)
        assert False
    except ValueError as e:
        assert "not within the required interval" in str(e)

    try:
        img = "../data/dataset/apple.jpg"
        _ = C.Pad(padding=4)(img)
        assert False
    except TypeError as e:
        assert "Input should be NumPy or PIL image" in str(e)


def test_eager_exceptions_normalize():
    """
    Feature: Normalize op
    Description: Exception eager support test for Normalize Python op
    Expectation: Error input image is detected
    """
    try:
        img = Image.open("../data/dataset/apple.jpg").convert("RGB")
        mean_vec = [.100, .100, .100]
        std_vec = [.2, .2, .2]
        _ = PY.Normalize(mean=mean_vec, std=std_vec)(img)
        assert False
    except TypeError as e:
        assert "img should be NumPy image" in str(e)


def test_eager_exceptions_pad():
    """
    Feature: Pad op
    Description: Exception eager support test for Pad Python op
    Expectation: Error input image is detected
    """
    try:
        img = "../data/dataset/apple.jpg"
        _ = PY.Pad(padding=4)(img)
        assert False
    except TypeError as e:
        assert "img should be PIL image" in str(e)


def test_eager_invalid_image_randomadjustsharpness_c():
    """
    Feature: RandomAdjustSharpness op
    Description: Exception eager support test for invalid image type
    Expectation: Error input image is detected
    """

    def test_config(my_input, error_type, error_msg):
        with pytest.raises(error_type) as error_info:
            _ = C.RandomAdjustSharpness(degree=0.5, prob=1)(my_input)
        assert error_msg in str(error_info.value)

    test_config((10,), TypeError, "Input should be NumPy or PIL image, got <class 'tuple'>")
    test_config(10, TypeError, "Input should be NumPy or PIL image, got <class 'int'>")


def test_eager_invalid_image_hwc2chw_c():
    """
    Feature: HWC2CHW op
    Description: Exception eager support test for HWC2CHW C++ op with invalid image input type
    Expectation: Error input image is detected
    """

    def test_config(my_input, error_type, error_msg):
        with pytest.raises(error_type) as error_info:
            _ = C.HWC2CHW()(my_input)
        assert error_msg in str(error_info.value)

    my_input = np.random.randn(64, 32, 3).astype(np.int32).tolist()
    test_config(my_input, TypeError, "Input should be NumPy or PIL image, got <class 'list'>")


def test_eager_invalid_image_hwc2chw_py():
    """
    Feature: HWC2CHW op
    Description: Exception eager support test for HWC2CHW Python op with invalid image input type
    Expectation: Error input image is detected
    """

    def test_config(my_input, error_type, error_msg):
        with pytest.raises(error_type) as error_info:
            _ = PY.HWC2CHW()(my_input)
        assert error_msg in str(error_info.value)

    my_input = np.random.randn(64, 32, 3).astype(np.int32).tolist()
    test_config(my_input, TypeError, "img should be NumPy array. Got <class 'list'>")


def test_eager_invalid_image_invert_c():
    """
    Feature: Invert op
    Description: Exception eager support test for Invert C++ op with invalid image type
    Expectation: Error input image is detected
    """

    def test_config(my_input, error_type, error_msg):
        with pytest.raises(error_type) as error_info:
            _ = C.Invert()(my_input)
        assert error_msg in str(error_info.value)

    test_config((10,), TypeError, "Input should be NumPy or PIL image, got <class 'tuple'>")
    test_config(10, TypeError, "Input should be NumPy or PIL image, got <class 'int'>")


def test_eager_invalid_image_pad_c():
    """
    Feature: Pad op
    Description: Exception eager support test for Pad C++ op with invalid image input type
    Expectation: Error input image is detected
    """

    def test_config(my_input, error_type, error_msg):
        with pytest.raises(error_type) as error_info:
            _ = C.Pad(padding=10)(my_input)
        assert error_msg in str(error_info.value)

    my_input = np.random.randn(64, 32, 3).astype(np.int32).tolist()
    test_config(my_input, TypeError, "Input should be NumPy or PIL image, got <class 'list'>")


def test_eager_invalid_image_randomcrop_c():
    """
    Feature: RandomCrop op
    Description: Exception eager support test for RandomCrop C++ op with invalid image input type
    Expectation: Error input image is detected
    """

    def test_config(my_input, error_type, error_msg):
        with pytest.raises(error_type) as error_info:
            _ = C.RandomCrop(size=200)(my_input)
        assert error_msg in str(error_info.value)

    my_input = np.random.randint(0, 255, (987, 654, 3)).astype(np.uint8).tolist()
    test_config(my_input, TypeError, "Input should be NumPy or PIL image, got <class 'list'>")


def test_eager_invalid_image_randomcrop_py():
    """
    Feature: RandomCrop op
    Description: Exception eager support test for RandomCrop Python op with invalid image input type
    Expectation: Error input image is detected
    """

    def test_config(my_input, error_type, error_msg):
        with pytest.raises(error_type) as error_info:
            _ = PY.RandomCrop(size=200)(my_input)
        assert error_msg in str(error_info.value)

    my_input = np.random.randint(0, 255, (987, 654, 3)).astype(np.uint8).tolist()
    test_config(my_input, TypeError, "img should be PIL image. Got <class 'list'>")


def test_eager_invalid_image_randomhorizontalflip_c():
    """
    Feature: RandomHorizontalFlip op
    Description: Exception eager support test for RandomHorizontalFlip C++ op with invalid image input type
    Expectation: Error input image is detected
    """

    def test_config(my_input, error_type, error_msg):
        with pytest.raises(error_type) as error_info:
            _ = C.RandomHorizontalFlip(prob=1)(my_input)
        assert error_msg in str(error_info.value)

    img = cv2.imread("../data/dataset/apple.jpg")
    my_input = img.tolist()
    test_config(my_input, TypeError, "Input should be NumPy or PIL image, got <class 'list'>")


def test_eager_invalid_image_randomsolarize_c():
    """
    Feature: RandomSolarize op
    Description: Exception eager support test for RandomSolarize C++ op with invalid image input type
    Expectation: Error input image is detected
    """

    def test_config(my_input, error_type, error_msg):
        with pytest.raises(error_type) as error_info:
            _ = C.RandomSolarize(threshold=(0, 120))(my_input)
        assert error_msg in str(error_info.value)

    my_input = np.random.randint(0, 255, (500, 600, 3)).astype(np.uint8).tolist()
    test_config(my_input, TypeError, "Input should be NumPy or PIL image, got <class 'list'>")


def test_eager_invalid_image_cutout_c():
    """
    Feature: CutOut op
    Description: Exception eager support test for CutOut C++ op with invalid image input type
    Expectation: Error input image is detected
    """

    def test_config(my_input, error_type, error_msg):
        with pytest.raises(error_type) as error_info:
            _ = C.CutOut(length=120, num_patches=1)(my_input)
        assert error_msg in str(error_info.value)

    my_input = np.random.randn(60, 50)
    test_config(my_input, RuntimeError, "CutOut: shape is invalid.")

    test_config(1, TypeError, "Input should be NumPy or PIL image, got <class 'int'>.")
    test_config(1.0, TypeError, "Input should be NumPy or PIL image, got <class 'float'>.")
    test_config((10, 20), TypeError, "Input should be NumPy or PIL image, got <class 'tuple'>.")
    test_config([10, 20, 30], TypeError, "Input should be NumPy or PIL image, got <class 'list'>.")


if __name__ == '__main__':
    test_eager_decode_c()
    test_eager_decode_py()
    test_eager_resize_c()
    test_eager_resize_py()
    test_eager_rescale()
    test_eager_normalize_c()
    test_eager_normalize_py()
    test_eager_resize_totensor_normalize_py()
    test_eager_compose_py()
    test_eager_hwc2chw()
    test_eager_pad_c()
    test_eager_pad_py()
    test_eager_cutout_pil_c()
    test_eager_cutout_pil_py()
    test_eager_cutout_cv_c()
    test_eager_exceptions()
    test_eager_exceptions_normalize()
    test_eager_exceptions_pad()
    test_eager_invalid_image_randomadjustsharpness_c()
    test_eager_invalid_image_hwc2chw_c()
    test_eager_invalid_image_hwc2chw_py()
    test_eager_invalid_image_invert_c()
    test_eager_invalid_image_pad_c()
    test_eager_invalid_image_randomcrop_c()
    test_eager_invalid_image_randomcrop_py()
    test_eager_invalid_image_randomhorizontalflip_c()
    test_eager_invalid_image_randomsolarize_c()
    test_eager_invalid_image_cutout_c()
