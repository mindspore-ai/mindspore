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
Testing PadToSize.
"""
import cv2
import numpy as np
from PIL import Image
import pytest

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore.dataset.vision import Border, ConvertMode

IMAGE_DIR = "../data/dataset/testPK/data"
CIFAR10_DIR = "../data/dataset/testCifar10Data"


def test_pad_to_size_size():
    """
    Feature: PadToSize
    Description: Test parameter `size`
    Expectation: Output image shape is as expected
    """
    dataset = ds.ImageFolderDataset(IMAGE_DIR, num_samples=10)
    transforms = [vision.Decode(to_pil=False),
                  vision.PadToSize(5000)]
    dataset = dataset.map(operations=transforms, input_columns=["image"])
    for data in dataset.create_dict_iterator(num_epochs=1):
        assert data["image"].shape == (5000, 5000, 3)

    dataset = ds.ImageFolderDataset(IMAGE_DIR, num_samples=10)
    transforms = [vision.Decode(to_pil=True),
                  vision.PadToSize((2500, 4500))]
    dataset = dataset.map(operations=transforms, input_columns=["image"])
    for data in dataset.create_dict_iterator(num_epochs=1):
        assert data["image"].shape == (2500, 4500, 3)


def test_pad_to_size_offset():
    """
    Feature: PadToSize
    Description: Test parameter `offset`
    Expectation: Output image shape is as expected
    """
    dataset = ds.Cifar10Dataset(CIFAR10_DIR, num_samples=10, shuffle=False)
    transforms = [vision.PadToSize((61, 57), None)]  # offset = None
    dataset = dataset.map(operations=transforms, input_columns=["image"])
    for data in dataset.create_dict_iterator(num_epochs=1):
        assert data["image"].shape == (61, 57, 3)

    dataset = ds.Cifar10Dataset(CIFAR10_DIR, num_samples=10, shuffle=False)
    transforms = [vision.PadToSize((61, 57), ())]  # offset is empty
    dataset = dataset.map(operations=transforms, input_columns=["image"])
    for data in dataset.create_dict_iterator(num_epochs=1):
        assert data["image"].shape == (61, 57, 3)

    dataset = ds.Cifar10Dataset(CIFAR10_DIR, num_samples=10, shuffle=False)
    transforms = [vision.PadToSize((61, 57), 5)]  # offset is int
    dataset = dataset.map(operations=transforms, input_columns=["image"])
    for data in dataset.create_dict_iterator(num_epochs=1):
        assert data["image"].shape == (61, 57, 3)

    dataset = ds.Cifar10Dataset(CIFAR10_DIR, num_samples=10, shuffle=False)
    transforms = [vision.PadToSize((61, 57), (3, 7))]  # offset is sequence
    dataset = dataset.map(operations=transforms, input_columns=["image"])
    for data in dataset.create_dict_iterator(num_epochs=1):
        assert data["image"].shape == (61, 57, 3)


def test_pad_to_size_eager(show=False):
    """
    Feature: PadToSize
    Description: Test eager mode
    Expectation: Output image shape is as expected
    """
    img = cv2.imread("../data/dataset/apple.jpg")
    img = vision.PadToSize(size=(3500, 7000), offset=None, fill_value=255, padding_mode=Border.EDGE)(img)
    assert img.shape == (3500, 7000, 3)

    img = Image.open("../data/dataset/apple.jpg").convert("RGB")
    img = vision.PadToSize(size=(3500, 7000), fill_value=(0, 0, 255), padding_mode=Border.CONSTANT)(img)
    assert img.shape == (3500, 7000, 3)
    if show:
        Image.fromarray(img).show()


def test_pad_to_size_grayscale():
    """
    Feature: PadToSize
    Description: Test on grayscale image
    Expectation: Output image shape is as expected
    """
    dataset = ds.Cifar10Dataset(CIFAR10_DIR, num_samples=10, shuffle=False)
    transforms = [vision.ConvertColor(ConvertMode.COLOR_RGB2GRAY),
                  vision.PadToSize(97)]
    dataset = dataset.map(operations=transforms, input_columns=["image"])
    for data in dataset.create_dict_iterator(num_epochs=1):
        assert data["image"].shape == (97, 97)


def test_pad_to_size_vs_pad():
    """
    Feature: PadToSize
    Description: Test the result comparing with Pad
    Expectation: Results of PadToSize and Pad are the same
    """
    original_size = (32, 32)

    dataset_pad_to_size = ds.Cifar10Dataset(CIFAR10_DIR, num_samples=10, shuffle=False)
    target_size = (50, 101)
    offset = (5, 13)
    transforms_pad_to_size = [vision.PadToSize(target_size, offset, fill_value=200, padding_mode=Border.CONSTANT)]
    dataset_pad_to_size = dataset_pad_to_size.map(operations=transforms_pad_to_size, input_columns=["image"])

    dataset_pad = ds.Cifar10Dataset(CIFAR10_DIR, num_samples=10, shuffle=False)
    left = offset[1]
    top = offset[0]
    right = target_size[1] - original_size[1] - left
    bottom = target_size[0] - original_size[0] - top
    transforms_pad = [vision.Pad((left, top, right, bottom), fill_value=200, padding_mode=Border.CONSTANT)]
    dataset_pad = dataset_pad.map(operations=transforms_pad, input_columns=["image"])

    for data_pad_to_size, data_pad in zip(dataset_pad_to_size.create_dict_iterator(num_epochs=1, output_numpy=True),
                                          dataset_pad.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(data_pad_to_size["image"], data_pad["image"])


def test_pad_to_size_check():
    """
    Feature: PadToSize
    Description: Test parameter check
    Expectation: Errors and logs are as expected
    """

    def test_invalid_input(error, error_msg, size=100, offset=None, fill_value=0, padding_mode=Border.CONSTANT,
                           data=np.random.random((28, 28, 3))):
        with pytest.raises(error) as error_info:
            _ = vision.PadToSize(size, offset, fill_value, padding_mode)(data)
        assert error_msg in str(error_info.value)

    # validate size
    test_invalid_input(TypeError, "is not of type", size=3.5)
    test_invalid_input(ValueError, "The size must be a sequence of length 2", size=())
    test_invalid_input(ValueError, "is not within the required interval", size=-100)
    test_invalid_input(ValueError, "is not within the required interval", size=(0, 50))

    # validate offset
    test_invalid_input(TypeError, "is not of type", offset="5")
    test_invalid_input(ValueError, "The offset must be empty or a sequence of length 2", offset=(5, 5, 5))
    test_invalid_input(ValueError, "is not within the required interval", offset=(-1, 10))

    # validate fill_value
    test_invalid_input(TypeError, "fill_value should be a single integer or a 3-tuple", fill_value=(0, 0))
    test_invalid_input(ValueError, "Input fill_value is not within the required interval", fill_value=-1)
    test_invalid_input(TypeError, "Argument fill_value[0] with value 100.0 is not of type", fill_value=(100.0, 10, 1))

    # validate padding_mode
    test_invalid_input(TypeError, "is not of type", padding_mode="CONSTANT")

    # validate data
    test_invalid_input(RuntimeError, "target size to pad should be no less than the original image size", size=(5, 5))
    test_invalid_input(RuntimeError, "sum of offset and original image size should be no more than the target size",
                       (30, 30), (5, 5))
    test_invalid_input(RuntimeError, "Expecting tensor in channel of (1, 3)",
                       data=np.random.random((28, 28, 4)))
    test_invalid_input(RuntimeError, "Expecting tensor in dimension of (2, 3)",
                       data=np.random.random(28))
    test_invalid_input(RuntimeError, "Expecting tensor in type of "
                                     "(bool, int8, uint8, int16, uint16, int32, float16, float32, float64)",
                       data=np.random.random((28, 28, 3)).astype(np.str))


if __name__ == "__main__":
    test_pad_to_size_size()
    test_pad_to_size_offset()
    test_pad_to_size_eager()
    test_pad_to_size_grayscale()
    test_pad_to_size_vs_pad()
    test_pad_to_size_check()
