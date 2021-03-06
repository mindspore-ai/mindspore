# Copyright 2020 Huawei Technologies Co., Ltd
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
from PIL import Image
import mindspore.dataset.vision.c_transforms as C
from mindspore import log as logger


def test_eager_decode():
    img = np.fromfile("../data/dataset/apple.jpg", dtype=np.uint8)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img = C.Decode()(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    assert img.shape == (2268, 4032, 3)


def test_eager_resize():
    img = cv2.imread("../data/dataset/apple.jpg")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    img = C.Resize(size=(32, 32))(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))

    assert img.shape == (32, 32, 3)

def test_eager_rescale():
    img = cv2.imread("../data/dataset/apple.jpg")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    pixel = img[0][0][0]

    rescale_factor = 0.5
    img = C.Rescale(rescale=rescale_factor, shift=0)(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    pixel_rescaled = img[0][0][0]

    assert pixel*rescale_factor == pixel_rescaled

def test_eager_normalize():
    img = Image.open("../data/dataset/apple.jpg").convert("RGB")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.size))
    pixel = img.getpixel((0, 0))[0]

    mean_vec = [100, 100, 100]
    std_vec = [2, 2, 2]
    img = C.Normalize(mean=mean_vec, std=std_vec)(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    pixel_normalized = img[0][0][0]

    assert (pixel - mean_vec[0]) / std_vec[0] == pixel_normalized

def test_eager_HWC2CHW():
    img = cv2.imread("../data/dataset/apple.jpg")
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    channel = img.shape

    img = C.HWC2CHW()(img)
    logger.info("Image.type: {}, Image.shape: {}".format(type(img), img.shape))
    channel_swaped = img.shape

    assert channel == (channel_swaped[1], channel_swaped[2], channel_swaped[0])

def test_eager_pad():
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

def test_eager_exceptions():
    try:
        img = "../data/dataset/apple.jpg"
        img = C.Decode()(img)
        assert False
    except TypeError as e:
        assert "Input should be an encoded image in 1-D NumPy format" in str(e)

    try:
        img = np.array(["a", "b", "c"])
        img = C.Decode()(img)
        assert False
    except TypeError as e:
        assert "Input should be an encoded image in 1-D NumPy format" in str(e)

    try:
        img = cv2.imread("../data/dataset/apple.jpg")
        img = C.Resize(size=(-32, 32))(img)
        assert False
    except ValueError as e:
        assert "not within the required interval" in str(e)

    try:
        img = "../data/dataset/apple.jpg"
        img = C.Pad(padding=4)(img)
        assert False
    except TypeError as e:
        assert "Input should be NumPy or PIL image" in str(e)


if __name__ == '__main__':
    test_eager_decode()
    test_eager_resize()
    test_eager_rescale()
    test_eager_normalize()
    test_eager_HWC2CHW()
    test_eager_pad()
    test_eager_exceptions()
  