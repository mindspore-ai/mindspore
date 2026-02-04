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
Testing PadToSize.
"""
import cv2
import numpy as np
import os
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as vision
from mindspore.dataset.vision import Border, ConvertMode

IMAGE_DIR = "../data/dataset/testPK/data"
CIFAR10_DIR = "../data/dataset/testCifar10Data"
TEST_DATA_DATASET_FUNC ="../data/dataset/"


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
                       data=np.random.random((28, 28, 3)).astype(np.str_))


def test_pad_to_size_operation_01():
    """
    Feature: PadToSize operation
    Description: Testing the normal functionality of the PadToSize operator
    Expectation: The Output is equal to the expected output
    """
    # PadToSize operator: Normal testing, The parameter is of type int.
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, 'test_data', 'testImageNetData2', 'train')
    dataset = ds.ImageFolderDataset(dataset_dir, num_samples=10, shuffle=False)
    transforms = [vision.Decode(),
                  vision.PadToSize(size=1000, offset=100, fill_value=100, padding_mode=Border.EDGE)]
    dataset = dataset.map(operations=transforms, input_columns=["image"])
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert data["image"].shape == (1000, 1000, 3)

    # PadToSize operator: Normal testing, The parameter is of type list.
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, 'test_data', 'testImageNetData2', 'train')
    dataset = ds.ImageFolderDataset(dataset_dir, num_samples=10, shuffle=False, decode=True)
    transforms = [vision.PadToSize(size=[1000, 900], offset=[100, 150], fill_value=(0, 0, 255),
                                   padding_mode=Border.CONSTANT)]
    dataset = dataset.map(operations=transforms, input_columns=["image"])
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert data["image"].shape == (1000, 900, 3)

    # PadToSize operator: Normal testing, The parameter is of type tuple.
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, 'test_data', 'testImageNetData2', 'train')
    dataset = ds.ImageFolderDataset(dataset_dir, num_samples=10, shuffle=False, decode=True)
    transforms = [vision.PadToSize(size=(1000, 900), offset=(100, 150), fill_value=100, padding_mode=Border.REFLECT)]
    dataset = dataset.map(operations=transforms, input_columns=["image"])
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert data["image"].shape == (1000, 900, 3)

    # PadToSize operator: Normal testing, Default parameters.
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, 'test_data', 'testImageNetData2', 'train')
    dataset1 = ds.ImageFolderDataset(dataset_dir, num_samples=10, shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(dataset_dir, num_samples=10, shuffle=False, decode=True)
    transforms1 = [vision.PadToSize(size=1000)]
    transforms2 = [vision.PadToSize(size=1000, offset=None, fill_value=0, padding_mode=Border.CONSTANT)]
    dataset1 = dataset1.map(operations=transforms1, input_columns=["image"])
    dataset2 = dataset2.map(operations=transforms2, input_columns=["image"])
    for data1, data2 in zip(dataset1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            dataset2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        assert data1["image"].shape == (1000, 1000, 3) == data2["image"].shape

    # PadToSize operator: Normal testing, input data is PIL image.
    img = Image.open(os.path.join(TEST_DATA_DATASET_FUNC, 'test_data', 'pen.jpg')).convert("RGB")
    img = vision.PadToSize(size=(1000, 1000), offset=None, fill_value=(0, 0, 255), padding_mode=Border.CONSTANT)(img)
    assert img.shape == (1000, 1000, 3)

    # PadToSize operator: Normal testing, In pipeline mode, the input data is a PIL image.
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, 'test_data', 'testImageNetData2', 'train')
    dataset = ds.ImageFolderDataset(dataset_dir, num_samples=10, shuffle=False)
    transforms = [vision.Decode(to_pil=True),
                  vision.PadToSize(size=1000)]
    dataset = dataset.map(operations=transforms, input_columns=["image"])
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert data["image"].shape == (1000, 1000, 3)

    # PadToSize operator: Normal testing, input data is numpy <H,W>
    dataset = np.random.randint(low=0, high=255, size=(10, 10, 10)).astype(np.uint8)
    dataset = ds.NumpySlicesDataset(dataset, column_names=["image"])
    transforms = [vision.PadToSize(size=20, offset=10)]
    dataset = dataset.map(operations=transforms, input_columns=["image"])
    num_iter = 0
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1
        assert data["image"].shape == (20, 20)
    assert num_iter == 10

    # PadToSize operator: Normal testing, input data is numpy <H,W> and fill_value=(100, 150, 200)
    data = np.random.randint(low=0, high=255, size=(10, 10)).astype(np.uint8)
    transforms = vision.PadToSize(size=20, offset=10, fill_value=(100, 150, 200))
    output = transforms(data)
    assert output.shape == (20, 20)

    # PadToSize operator: Normal testing, input data is numpy <H,W,1> and fill_value=100
    dataset = np.random.randint(low=0, high=255, size=(10, 10, 10, 1)).astype(np.uint8)
    dataset = ds.NumpySlicesDataset(dataset, column_names=["image"])
    transforms = [vision.PadToSize(size=20, offset=10, fill_value=100)]
    dataset = dataset.map(operations=transforms, input_columns=["image"])
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert data["image"].shape == (20, 20, 1)

    # PadToSize operator: Normal testing, input data is numpy <H,W,1> and fill_value=(100, 150, 200)
    data = np.random.randint(low=0, high=255, size=(10, 10, 1)).astype(np.uint8)
    transforms = vision.PadToSize(size=20, offset=10, fill_value=(100, 150, 200))
    output = transforms(data)
    assert output.shape == (20, 20, 1)


def test_pad_to_size_exception_01():
    """
    Feature: PadToSize operation
    Description: Testing the PadToSize Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # PadToSize Operator: Exception Testing, offset + original image dimensions exceed size
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, 'test_data', 'testImageNetData2', 'train')
    dataset = ds.ImageFolderDataset(dataset_dir, num_samples=10, shuffle=False, decode=True)
    with pytest.raises(RuntimeError, match=r"PadToSize: the sum of offset and original image size should be no "
                                           r"more than the target size to pad, but got offset \(100, 250\) plus "
                                           r"original size .* bigger than \(1000, 900\)"):
        transforms = [vision.PadToSize(size=[1000, 900], offset=[100, 250])]
        dataset = dataset.map(operations=transforms, input_columns=["image"])
        for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass

    # PadToSize Operator: Exception Testing, Size smaller than the original image dimensions
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, 'test_data', 'testImageNetData2', 'train')
    dataset = ds.ImageFolderDataset(dataset_dir, num_samples=10, shuffle=False, decode=True)
    with pytest.raises(RuntimeError, match=r"Syntax error"):
        transforms = [vision.PadToSize(size=10)]
        dataset = dataset.map(operations=transforms, input_columns=["image"])
        for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass

    # PadToSize Operator: Exception Testing, size data type error
    with pytest.raises(TypeError, match=r"Argument size with value 123 is not of type \[<class 'int'>, "
                                        r"<class 'list'>, <class 'tuple'>\], but got <class 'str'>."):
        vision.PadToSize(size='123')

    # PadToSize Operator: Exception Testing, size=0
    with pytest.raises(ValueError, match=r"Input size is not within the required interval of \[1, 2147483647\]."):
        vision.PadToSize(size=0)

    # PadToSize Operator: Exception Testing, size=2147483648
    with pytest.raises(ValueError, match=r"Input size is not within the required interval of \[1, 2147483647\]."):
        vision.PadToSize(size=2147483648)

    # PadToSize Operator: Exception Testing, The length of size is 3.
    with pytest.raises(ValueError, match=r"The size must be a sequence of length 2."):
        vision.PadToSize(size=[1000, 1000, 1000])

    # PadToSize Operator: Exception Testing, The length of size is 1.
    with pytest.raises(ValueError, match=r"The size must be a sequence of length 2."):
        vision.PadToSize(size=[1000])

    # PadToSize Operator: Exception Testing, Incorrect element type in size.
    with pytest.raises(TypeError, match=r"Argument size0 with value 1000.0 is not of "
                                        r"type \[<class 'int'>\], but got <class 'float'>."):
        vision.PadToSize(size=[1000.0, 1000])

    # PadToSize Operator: Exception Testing, The element value in size is 0.
    with pytest.raises(ValueError, match=r"Input size0 is not within the required interval of \[1, 2147483647\]."):
        vision.PadToSize(size=[0, 1000])

    # PadToSize Operator: Exception Testing, The element value in size is 2147483648.
    with pytest.raises(ValueError, match=r"Input size0 is not within the required interval of \[1, 2147483647\]."):
        vision.PadToSize(size=[2147483648, 1000])

    # PadToSize Operator: Exception Testing, offset type error
    with pytest.raises(TypeError, match=r"Argument offset with value 123 is not of type \[<class 'int'>, "
                                        r"<class 'list'>, <class 'tuple'>\], but got <class 'str'>."):
        vision.PadToSize(size=1000, offset='123')

    # PadToSize Operator: Exception Testing, offset=-1
    with pytest.raises(ValueError, match=r"Input offset is not within the required interval of \[0, 2147483647\]."):
        vision.PadToSize(size=1000, offset=-1)

    # PadToSize Operator: Exception Testing, offset=2147483648
    with pytest.raises(ValueError, match=r"Input offset is not within the required interval of \[0, 2147483647\]."):
        vision.PadToSize(size=1000, offset=2147483648)

    # PadToSize Operator: Exception Testing, The length of the offset is 3.
    with pytest.raises(ValueError, match=r"The offset must be empty or a sequence of length 2."):
        vision.PadToSize(size=1000, offset=[1000, 1000, 1000])

    # PadToSize Operator: Exception Testing, The length of the offset is 1.
    with pytest.raises(ValueError, match=r"The offset must be empty or a sequence of length 2."):
        vision.PadToSize(size=1000, offset=[1000])

    # PadToSize Operator: Exception Testing, Incorrect element type in offset.
    with pytest.raises(TypeError, match=r"Argument offset0 with value 1000.0 is not of "
                                        r"type \[<class 'int'>\], but got <class 'float'>."):
        vision.PadToSize(size=1000, offset=[1000.0, 1000])

    # PadToSize Operator: Exception Testing, The element value in offset is -1.
    with pytest.raises(ValueError, match=r"Input offset0 is not within the required interval of \[0, 2147483647\]."):
        vision.PadToSize(size=1000, offset=[-1, 1000])

    # PadToSize Operator: Exception Testing, The element value in offset is 2147483648.
    with pytest.raises(ValueError, match=r"Input offset0 is not within the required interval of \[0, 2147483647\]."):
        vision.PadToSize(size=1000, offset=[2147483648, 1000])

    # PadToSize Operator: Exception Testing, fill_value type error
    with pytest.raises(TypeError, match=r"fill_value should be a single integer or a 3-tuple."):
        vision.PadToSize(size=1000, fill_value='123')

    # PadToSize Operator: Exception Testing, fill_value=-1
    with pytest.raises(ValueError, match=r"Input fill_value is not within the required interval of \[0, 255\]."):
        vision.PadToSize(size=1000, fill_value=-1)

    # PadToSize Operator: Exception Testing, fill_value=256
    with pytest.raises(ValueError, match=r"Input fill_value is not within the required interval of \[0, 255\]."):
        vision.PadToSize(size=1000, fill_value=256)

    # PadToSize Operator: Exception Testing, The length of the fill_value is 4.
    with pytest.raises(TypeError, match=r"fill_value should be a single integer or a 3-tuple."):
        vision.PadToSize(size=1000, fill_value=(100, 100, 100, 100))

    # PadToSize Operator: Exception Testing, The length of the fill_value is 1.
    with pytest.raises(TypeError, match=r"fill_value should be a single integer or a 3-tuple."):
        vision.PadToSize(size=1000, fill_value=(100,))


def test_pad_to_size_exception_02():
    """
    Feature: PadToSize operation
    Description: Testing the PadToSize Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # PadToSize Operator: Exception Testing, Incorrect element type in fill_value
    with pytest.raises(TypeError, match=r"Argument fill_value\[0] with value 100.0 is not of "
                                        r"type \[<class 'int'>\], but got <class 'float'>."):
        vision.PadToSize(size=1000, fill_value=(100.0, 100, 100))

    # PadToSize Operator: Exception Testing, The element value in fill_value is 1.
    with pytest.raises(ValueError, match=r"Input fill_value\[0\] is not within the required interval of \[0, 255\]."):
        vision.PadToSize(size=1000, fill_value=(-1, 100, 100))

    # PadToSize Operator: Exception Testing, The element value in fill_value is 256.
    with pytest.raises(ValueError, match=r"Input fill_value\[0\] is not within the required interval of \[0, 255\]."):
        vision.PadToSize(size=1000, fill_value=(256, 100, 100))

    # PadToSize Operator: Exception Testing, padding_mode type error
    with pytest.raises(TypeError, match=r"Argument padding_mode with value 123 is not of "
                                        r"type \[<enum 'Border'>\], but got <class 'str'>."):
        vision.PadToSize(size=1000, padding_mode='123')

    # PadToSize Operator: Exception Testing, Input data is a list whose elements are floats.
    img = [[1, 2, 3], [4, 5, 6], [7, 8, 9.0]]
    op = vision.PadToSize(size=100)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        op(img)

    # PadToSize Operator: Exception Testing, Input data is a list whose elements are int.
    img = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    op = vision.PadToSize(size=100)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        op(img)

    # PadToSize Operator: Exception Testing, Input data is a list whose elements are str.
    img = [[[1, 2],
            [3, 4]],
           [[1, 2],
            ['3', 2.0]]]
    op = vision.PadToSize(size=10)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        op(img)

    # PadToSize Operator: Exception Testing, input data is numpy <H,W,2>
    data = np.random.randint(low=0, high=255, size=(10, 10, 2)).astype(np.uint8)
    transforms = vision.PadToSize(size=20, offset=10, fill_value=100)
    with pytest.raises(RuntimeError,
                       match=r"PadToSize: the channel of image tensor does not "
                             r"match the requirement of operator. Expecting tensor in channel "
                             r"of \(1, 3\). But got channel 2."):
        transforms(data)

    # PadToSize Operator: Exception Testing, input data is numpy <H,W,4>
    data = np.random.randint(low=0, high=255, size=(10, 10, 4)).astype(np.uint8)
    transforms = vision.PadToSize(size=20, fill_value=(100, 150, 200))
    with pytest.raises(RuntimeError,
                       match=r"PadToSize: the channel of image tensor does not "
                             r"match the requirement of operator. Expecting tensor in channel "
                             r"of \(1, 3\). But got channel 4."):
        transforms(data)

    # PadToSize Operator: Exception Testing, input data is numpy <H,W,5>
    data = np.random.randint(low=0, high=255, size=(10, 10, 5)).astype(np.uint8)
    transforms = vision.PadToSize(size=20, offset=10, fill_value=100)
    with pytest.raises(RuntimeError,
                       match=r"PadToSize: the channel of image tensor does not "
                             r"match the requirement of operator. Expecting tensor in channel "
                             r"of \(1, 3\). But got channel 5."):
        transforms(data)

    # PadToSize Operator: Exception Testing, Input data is one-dimensional data in NumPy format.
    data = np.random.randint(low=0, high=255, size=(30,)).astype(np.float32)
    with pytest.raises(RuntimeError,
                       match=r"PadToSize: the dimension of image tensor does not "
                             r"match the requirement of operator. Expecting tensor in dimension "
                             r"of \(2, 3\), in shape of <H, W> or <H, W, C>. But got dimension 1. "
                             r"You may need to perform Decode first."):
        transforms = vision.PadToSize(size=20, fill_value=(100, 150, 200))
        transforms(data)

    # PadToSize Operator: Exception Testing, Input data is four-dimensional data in NumPy format.
    data = np.random.randint(low=0, high=255, size=(10, 10, 2, 2)).astype(np.uint8)
    transforms = vision.PadToSize(size=20, fill_value=(100, 150, 200))
    with pytest.raises(RuntimeError,
                       match=r"PadToSize: the dimension of image tensor does not "
                             r"match the requirement of operator. Expecting tensor in dimension "
                             r"of \(2, 3\), in shape of <H, W> or <H, W, C>. But got dimension 4."):
        transforms(data)

    # PadToSize Operator: Exception Testing, padding_mode=Border.EDGE
    data = np.array([['1', '2', '3'], ['4', '5', '6']], dtype=np.str_)
    transforms = vision.PadToSize(size=20, padding_mode=Border.EDGE)
    with pytest.raises(RuntimeError,
                       match=r"PadToSize: the data type of image tensor does not "
                             r"match the requirement of operator. Expecting tensor in type "
                             r"of \(bool, int8, uint8, int16, uint16, int32, float16, float32, "
                             r"float64\). But got type string."):
        transforms(data)

    # PadToSize Operator: Exception Testing, padding_mode=Border.SYMMETRIC
    data = np.array([['1', '2', '3'], ['4', '5', '6']], dtype=np.str_)
    transforms = vision.PadToSize(size=20, padding_mode=Border.SYMMETRIC)
    with pytest.raises(RuntimeError,
                       match=r"PadToSize: the data type of image tensor does not "
                             r"match the requirement of operator. Expecting tensor in type "
                             r"of \(bool, int8, uint8, int16, uint16, int32, float16, float32, "
                             r"float64\). But got type string."):
        transforms(data)


if __name__ == "__main__":
    test_pad_to_size_size()
    test_pad_to_size_offset()
    test_pad_to_size_eager()
    test_pad_to_size_grayscale()
    test_pad_to_size_vs_pad()
    test_pad_to_size_check()
    test_pad_to_size_operation_01()
    test_pad_to_size_exception_01()
    test_pad_to_size_exception_02()
