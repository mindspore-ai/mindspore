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
import cv2
import numpy as np
import pytest
from PIL import Image
import mindspore.dataset.vision as vision
from mindspore import log as logger


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

    fp = open("../data/dataset/apple.jpg", "rb")
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

    fp = open("../data/dataset/apple.jpg", "rb")
    img2 = fp.read()

    img2 = vision.Decode(to_pil=True)(img2)
    logger.info("Image.type: {}, Image.shape: {}".format(
        type(img2), img2.size))
    assert img2.size == (4032, 2268)


def test_eager_resize():
    """
    Feature: Resize op
    Description: Test eager support for Resize op
    Expectation: Output image size from op is correct
    """
    img = cv2.imread("../data/dataset/apple.jpg")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img = vision.Resize(size=(32, 32))(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    assert img.shape == (32, 32, 3)


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
    assert "Input should be an encoded image in 1-D NumPy format" in str(error_info.value)

    with pytest.raises(TypeError) as error_info:
        img = np.array(["a", "b", "c"])
        _ = vision.Decode()(img)
    assert "Input should be an encoded image in 1-D NumPy format" in str(error_info.value)


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
    Feature: Pad op
    Description: Exception eager support test for Pad Python implementation
    Expectation: Error input image is detected
    """
    try:
        img = "../data/dataset/apple.jpg"
        _ = vision.Pad(padding=4)(img)
        assert False
    except RuntimeError as e:
        assert "tensor should be in shape of <H,W,C> or <H,W>" in str(e)


if __name__ == '__main__':
    test_eager_decode_c()
    test_eager_decode_py()
    test_eager_resize()
    test_eager_rescale()
    test_eager_normalize_hwc()
    test_eager_normalize_chw()
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
