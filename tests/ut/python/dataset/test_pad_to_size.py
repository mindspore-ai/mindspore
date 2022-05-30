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
import pytest

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore.dataset.vision import Border, ConvertMode

DATA_DIR_10 = "../data/dataset/testCifar10Data"


def test_pad_to_size_size():
    """
    Feature: PadToSize
    Description: Test parameter `size`
    Expectation: Output image shape is as expected
    """
    dataset = ds.Cifar10Dataset(DATA_DIR_10, num_samples=10, shuffle=False)
    transforms = [vision.PadToSize(100)]
    dataset = dataset.map(operations=transforms, input_columns=["image"])
    for data in dataset.create_dict_iterator(num_epochs=1):
        assert data["image"].shape == (100, 100, 3)

    dataset = ds.Cifar10Dataset(DATA_DIR_10, num_samples=10, shuffle=False)
    transforms = [vision.PadToSize((52, 66))]
    dataset = dataset.map(operations=transforms, input_columns=["image"])
    for data in dataset.create_dict_iterator(num_epochs=1):
        assert data["image"].shape == (52, 66, 3)


def test_pad_to_size_offset():
    """
    Feature: PadToSize
    Description: Test parameter `offset`
    Expectation: Output image shape is as expected
    """
    dataset = ds.Cifar10Dataset(DATA_DIR_10, num_samples=10, shuffle=False)
    transforms = [vision.PadToSize((61, 57), None)]  # offset = None
    dataset = dataset.map(operations=transforms, input_columns=["image"])
    for data in dataset.create_dict_iterator(num_epochs=1):
        assert data["image"].shape == (61, 57, 3)

    dataset = ds.Cifar10Dataset(DATA_DIR_10, num_samples=10, shuffle=False)
    transforms = [vision.PadToSize((61, 57), ())]  # offset is empty
    dataset = dataset.map(operations=transforms, input_columns=["image"])
    for data in dataset.create_dict_iterator(num_epochs=1):
        assert data["image"].shape == (61, 57, 3)

    dataset = ds.Cifar10Dataset(DATA_DIR_10, num_samples=10, shuffle=False)
    transforms = [vision.PadToSize((61, 57), 5)]  # offset is int
    dataset = dataset.map(operations=transforms, input_columns=["image"])
    for data in dataset.create_dict_iterator(num_epochs=1):
        assert data["image"].shape == (61, 57, 3)

    dataset = ds.Cifar10Dataset(DATA_DIR_10, num_samples=10, shuffle=False)
    transforms = [vision.PadToSize((61, 57), (3, 7))]  # offset is sequence
    dataset = dataset.map(operations=transforms, input_columns=["image"])
    for data in dataset.create_dict_iterator(num_epochs=1):
        assert data["image"].shape == (61, 57, 3)


def test_pad_to_size_eager():
    """
    Feature: PadToSize
    Description: Test eager mode
    Expectation: Output image shape is as expected
    """
    img = cv2.imread("../data/dataset/apple.jpg")
    img = vision.PadToSize(size=(3500, 7000), offset=None, fill_value=255, padding_mode=Border.EDGE)(img)
    assert img.shape == (3500, 7000, 3)


def test_pad_to_size_grayscale():
    """
    Feature: PadToSize
    Description: Test on grayscale image
    Expectation: Output image shape is as expected
    """
    dataset = ds.Cifar10Dataset(DATA_DIR_10, num_samples=10, shuffle=False)
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

    dataset_pad_to_size = ds.Cifar10Dataset(DATA_DIR_10, num_samples=10, shuffle=False)
    target_size = (50, 101)
    offset = (5, 13)
    transforms_pad_to_size = [vision.PadToSize(target_size, offset, fill_value=200, padding_mode=Border.CONSTANT)]
    dataset_pad_to_size = dataset_pad_to_size.map(operations=transforms_pad_to_size, input_columns=["image"])

    dataset_pad = ds.Cifar10Dataset(DATA_DIR_10, num_samples=10, shuffle=False)
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

    def test_invalid_input(error, error_msg, size, offset=None, fill_value=0, padding_mode=Border.CONSTANT):
        with pytest.raises(error) as error_info:
            _ = vision.PadToSize(size, offset, fill_value, padding_mode)(np.random.random((28, 28, 3)))
        assert error_msg in str(error_info.value)

    test_invalid_input(TypeError, "is not of type", 3.5)
    test_invalid_input(ValueError, "The size must be a sequence of length 2", ())
    test_invalid_input(ValueError, "must be greater than 0", -100)
    test_invalid_input(ValueError, "is not within the required interval", (0, 50))

    test_invalid_input(TypeError, "is not of type", 100, "5")
    test_invalid_input(ValueError, "The offset must be empty or a sequence of length 2", 100, (5, 5, 5))
    test_invalid_input(ValueError, "is not within the required interval", 100, (-1, 10))

    test_invalid_input(TypeError, "fill_value should be a single integer or a 3-tuple", 100, 10, (0, 0))
    test_invalid_input(ValueError, "is not within the required interval", 100, 10, -1)

    test_invalid_input(TypeError, "is not of type", 100, 10, 0, "CONSTANT")

    test_invalid_input(RuntimeError, "the target size to pad should be no less than the original image size", (5, 5))
    test_invalid_input(RuntimeError,
                       "the sum of offset and original image size should be no more than the target size to pad",
                       (30, 30), (5, 5))


if __name__ == "__main__":
    test_pad_to_size_size()
    test_pad_to_size_offset()
    test_pad_to_size_eager()
    test_pad_to_size_grayscale()
    test_pad_to_size_vs_pad()
    test_pad_to_size_check()
