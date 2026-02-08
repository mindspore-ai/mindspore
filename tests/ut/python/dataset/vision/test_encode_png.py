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
Testing encode_png
"""
import cv2
import numpy
import os
import pytest

import mindspore
import mindspore.dataset.vision.utils as v_trans

TEST_DATA_DATASET_FUNC ="../data/dataset/"


def test_encode_png_three_channels():
    """
    Feature: encode_png
    Description: Test encode_png by encoding the three channels image as PNG data according to the compression_level
    Expectation: Output is equal to the expected output
    """
    filename = "../data/dataset/apple.jpg"
    mode = cv2.IMREAD_UNCHANGED
    image = cv2.imread(filename, mode)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Test with numpy:ndarray and default compression_level
    encoded_png = mindspore.dataset.vision.encode_png(image_rgb)
    assert encoded_png.dtype == numpy.uint8
    assert encoded_png[0] == 137
    assert encoded_png[1] == 80
    assert encoded_png[2] == 78
    assert encoded_png[3] == 71

    # Test with Tensor and compression_level
    input_tensor = mindspore.Tensor.from_numpy(image_rgb)
    encoded_png_6 = mindspore.dataset.vision.encode_png(input_tensor, 6)
    assert encoded_png_6[1] == 80

    # Test with the minimum compression_level
    encoded_png_0 = mindspore.dataset.vision.encode_png(input_tensor, 0)
    assert encoded_png_0[1] == 80

    # Test with the maximum compression_level
    encoded_png_9 = mindspore.dataset.vision.encode_png(input_tensor, 9)
    assert encoded_png_9[1] == 80

    # Test with three channels 12*34*3 random uint8
    image_random = numpy.ndarray(shape=(12, 34, 3), dtype=numpy.uint8)
    encoded_png = mindspore.dataset.vision.encode_png(image_random)
    assert encoded_png[1] == 80
    encoded_png = mindspore.dataset.vision.encode_png(mindspore.Tensor.from_numpy(image_random))
    assert encoded_png[1] == 80


def test_encode_png_one_channel():
    """
    Feature: encode_png
    Description: Test encode_png by encoding the one channel image as PNG data
    Expectation: Output is equal to the expected output
    """
    filename = "../data/dataset/apple.jpg"
    mode = cv2.IMREAD_UNCHANGED
    image = cv2.imread(filename, mode)

    # Test with one channel image_grayscale
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    encoded_png = mindspore.dataset.vision.encode_png(image_grayscale)
    assert encoded_png[1] == 80
    encoded_png = mindspore.dataset.vision.encode_png(mindspore.Tensor.from_numpy(image_grayscale))
    assert encoded_png[1] == 80

    # Test with one channel 12*34 random uint8
    image_random = numpy.ndarray(shape=(12, 34), dtype=numpy.uint8)
    encoded_png = mindspore.dataset.vision.encode_png(image_random)
    assert encoded_png[1] == 80
    encoded_png = mindspore.dataset.vision.encode_png(mindspore.Tensor.from_numpy(image_random))
    assert encoded_png[1] == 80

    # Test with one channel 12*34*1 random uint8
    image_random = numpy.ndarray(shape=(12, 34, 1), dtype=numpy.uint8)
    encoded_png = mindspore.dataset.vision.encode_png(image_random)
    assert encoded_png[1] == 80
    encoded_png = mindspore.dataset.vision.encode_png(mindspore.Tensor.from_numpy(image_random))
    assert encoded_png[1] == 80


def test_encode_png_exception():
    """
    Feature: encode_png
    Description: Test encode_png with invalid parameter
    Expectation: Error is caught when the parameter is invalid
    """

    def test_invalid_param(image_param, compression_level_param, error, error_msg):
        """
        a function used for checking correct error and message with invalid parameter
        """
        with pytest.raises(error) as error_info:
            mindspore.dataset.vision.encode_png(image_param, compression_level_param)
        assert error_msg in str(error_info.value)

    filename = "../data/dataset/apple.jpg"
    mode = cv2.IMREAD_UNCHANGED
    image = cv2.imread(filename, mode)

    # Test with an invalid integer for the compression_level
    error_message = "Invalid compression_level"
    test_invalid_param(image, -1, RuntimeError, error_message)
    test_invalid_param(image, 10, RuntimeError, error_message)

    # Test with an invalid type for the compression_level
    error_message = "Input compression_level is not of type"
    test_invalid_param(image, 6.0, TypeError, error_message)

    # Test with an invalid image containing the float elements
    invalid_image = numpy.ndarray(shape=(10, 10, 3), dtype=float)
    error_message = "The type of the image data"
    test_invalid_param(invalid_image, 6, RuntimeError, error_message)

    # Test with an invalid type for the image
    error_message = "Input image is not of type"
    test_invalid_param("invalid_image", 6, TypeError, error_message)

    # Test with an invalid image with only one dimension
    invalid_image = numpy.ndarray(shape=(10), dtype=numpy.uint8)
    error_message = "The image has invalid dimensions"
    test_invalid_param(invalid_image, 6, RuntimeError, error_message)

    # Test with an invalid image with four dimensions
    invalid_image = numpy.ndarray(shape=(10, 10, 10, 3), dtype=numpy.uint8)
    test_invalid_param(invalid_image, 6, RuntimeError, error_message)

    # Test with an invalid image with two channels
    invalid_image = numpy.ndarray(shape=(10, 10, 2), dtype=numpy.uint8)
    error_message = "The image has invalid channels"
    test_invalid_param(invalid_image, 6, RuntimeError, error_message)


def test_encode_png_operation_01():
    """
    Feature: encode_png operation
    Description: Testing the normal functionality of the encode_png operator
    Expectation: The Output is equal to the expected output
    """
    # use default compression_level value 6, numpy type, jpg photo
    mode = cv2.IMREAD_UNCHANGED
    image_dir = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    image_numpy = cv2.imread(image_dir, mode)
    assert image_numpy.shape[2] == 3
    image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
    assert isinstance(image_numpy, numpy.ndarray)
    encode_png = v_trans.encode_png(image_numpy)
    assert isinstance(encode_png, numpy.ndarray)
    assert encode_png.dtype == 'uint8'
    assert encode_png.shape == (412135,)

    # use default compression_level value 6, tensor type,jpg photo
    mode = cv2.IMREAD_UNCHANGED
    image_dir = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    image_numpy = cv2.imread(image_dir, mode)
    assert image_numpy.shape[2] == 3
    image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
    assert isinstance(image_numpy, numpy.ndarray)
    image_tensor = mindspore.Tensor.from_numpy(image_numpy)
    encode_png = v_trans.encode_png(image_tensor, 6)
    assert isinstance(encode_png, numpy.ndarray)
    assert encode_png.dtype == 'uint8'
    assert encode_png.shape == (412135,)

    # use default compression_level value 0, tensor type, bmp photo
    mode = cv2.IMREAD_UNCHANGED
    data_dir4 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
    image_numpy = cv2.imread(data_dir4, mode)
    assert image_numpy.shape[2] == 3
    image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
    assert isinstance(image_numpy, numpy.ndarray)
    image_tensor = mindspore.Tensor.from_numpy(image_numpy)
    encode_png = v_trans.encode_png(image_tensor, 0)
    assert isinstance(encode_png, numpy.ndarray)
    assert encode_png.dtype == 'uint8'
    assert encode_png.shape == (34777,)

    # use default compression_level value 9, tensor type, png photo
    mode = cv2.IMREAD_UNCHANGED
    data_dir5 = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
    image_numpy = cv2.imread(data_dir5, mode)
    assert image_numpy.shape[2] == 4
    image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_BGRA2RGB)
    assert isinstance(image_numpy, numpy.ndarray)
    image_tensor = mindspore.Tensor.from_numpy(image_numpy)
    encode_png = v_trans.encode_png(image_tensor, 9)
    assert isinstance(encode_png, numpy.ndarray)
    assert encode_png.dtype == 'uint8'
    assert encode_png.shape == (427421,)

    # use default compression_level value 5, tensor type, jpg photo
    mode = cv2.IMREAD_UNCHANGED
    image_dir = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    image_numpy = cv2.imread(image_dir, mode)
    assert image_numpy.shape[2] == 3
    image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
    assert isinstance(image_numpy, numpy.ndarray)
    image_tensor = mindspore.Tensor.from_numpy(image_numpy)
    encode_png = v_trans.encode_png(image_tensor, 5)
    assert isinstance(encode_png, numpy.ndarray)
    assert encode_png.dtype == 'uint8'
    assert encode_png.shape == (412135,)

    # use default compression_level value 1, tensor type, jpg photo
    mode = cv2.IMREAD_UNCHANGED
    image_dir = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    image_numpy = cv2.imread(image_dir, mode)
    assert image_numpy.shape[2] == 3
    image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
    assert isinstance(image_numpy, numpy.ndarray)
    image_tensor = mindspore.Tensor.from_numpy(image_numpy)
    encode_png = v_trans.encode_png(image_tensor, 1)
    assert isinstance(encode_png, numpy.ndarray)
    assert encode_png.dtype == 'uint8'
    assert encode_png.shape == (412135,)

    # use default compression_level value 8, tensor type, jpg photo
    mode = cv2.IMREAD_UNCHANGED
    image_dir = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    image_numpy = cv2.imread(image_dir, mode)
    assert image_numpy.shape[2] == 3
    image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
    assert isinstance(image_numpy, numpy.ndarray)
    image_tensor = mindspore.Tensor.from_numpy(image_numpy)
    encode_png = v_trans.encode_png(image_tensor, 8)
    assert isinstance(encode_png, numpy.ndarray)
    assert encode_png.dtype == 'uint8'
    assert encode_png.shape == (412135,)

    # use random data, compression_level_2
    image_random = numpy.random.randint(256, size=(1, 1, 3), dtype=numpy.uint8)
    encode_png = mindspore.dataset.vision.encode_png(image_random, 2)
    assert isinstance(encode_png, numpy.ndarray)
    assert encode_png.dtype == 'uint8'
    assert encode_png.shape == (69,)


def test_encode_png_operation_02():
    """
    Feature: encode_png operation
    Description: Testing the normal functionality of the encode_png operator
    Expectation: The Output is equal to the expected output
    """
    # use random data, compression_level_2
    image_random = numpy.random.randint(256, size=(120, 340, 3), dtype=numpy.uint8)
    image_tensor = mindspore.Tensor.from_numpy(image_random)
    encode_png = mindspore.dataset.vision.encode_png(image_tensor, 4)
    assert isinstance(encode_png, numpy.ndarray)
    assert encode_png.dtype == 'uint8'
    assert encode_png.shape == (122791,)

    # use random data, compression_level_2
    image_random = numpy.random.randint(256, size=(1, 8000, 3), dtype=numpy.uint8)
    image_tensor = mindspore.Tensor.from_numpy(image_random)
    encode_png = mindspore.dataset.vision.encode_png(image_tensor, 3)
    assert isinstance(encode_png, numpy.ndarray)
    assert encode_png.dtype == 'uint8'
    assert encode_png.shape == (24098,)

    # use random data, compression_level_2
    image_random = numpy.random.randint(256, size=(600, 200, 3), dtype=numpy.uint8)
    image_tensor = mindspore.Tensor.from_numpy(image_random)
    encode_png = mindspore.dataset.vision.encode_png(image_tensor)
    assert isinstance(encode_png, numpy.ndarray)
    assert encode_png.dtype == 'uint8'
    assert encode_png.shape == (361306,)

    # use random data, compression_level_2
    image_random = numpy.random.randint(256, size=(876, 543), dtype=numpy.uint8)
    image_tensor = mindspore.Tensor.from_numpy(image_random)
    encode_png = mindspore.dataset.vision.encode_png(image_tensor, 7)
    assert isinstance(encode_png, numpy.ndarray)
    assert encode_png.dtype == 'uint8'
    assert encode_png.shape == (477453,)

    # use random data, compression_level_2
    image_random = numpy.random.randint(256, size=(224, 224, 1), dtype=numpy.uint8)
    image_tensor = mindspore.Tensor.from_numpy(image_random)
    encode_png = mindspore.dataset.vision.encode_png(image_tensor)
    assert isinstance(encode_png, numpy.ndarray)
    assert encode_png.dtype == 'uint8'
    assert encode_png.shape == (50555,)


def test_encode_png_exception_01():
    """
    Feature: encode_png operation
    Description: Testing the encode_png Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # use channels is not 3/1
    with pytest.raises(RuntimeError, match="EncodePng: The image has invalid "
                                           "channels. It should have 1 or 3 channels, but got 2 channels."):
        image_random = numpy.random.randint(256, size=(876, 543, 2), dtype=numpy.uint8)
        image_tensor = mindspore.Tensor.from_numpy(image_random)
        v_trans.encode_png(image_tensor)

    # use dimensions is not 3/2
    with pytest.raises(RuntimeError, match="EncodePng: The image has invalid "
                                           "dimensions. It should have two or three dimensions, but got 1 dimensions."):
        image_random = numpy.random.randint(256, size=(224,), dtype=numpy.uint8)
        image_tensor = mindspore.Tensor.from_numpy(image_random)
        v_trans.encode_png(image_tensor)

    # use dimensions is not 3/2
    try:
        image_random = numpy.random.randint(256, size=(224, 124, 1), dtype=numpy.uint16)
        v_trans.encode_png(image_random)
    except RuntimeError as e:
        assert "The type of the image data should be UINT8, but got uint16" in str(e)

    # use channels is not 3/1,
    try:
        image_random = numpy.random.randint(256, size=(876, 543, 1), dtype=numpy.uint8)
        mindspore.Tensor.from_numpy(image_random)
        v_trans.encode_png()
    except TypeError as e:
        assert "encode_png() missing 1 required positional argument: 'image'" in str(e)

    # compression_level_value_out_of_range
    with pytest.raises(RuntimeError, match=r"EncodePng: Invalid compression_level 10, should be in range of \[0, 9\]."):
        image_random = numpy.random.randint(256, size=(876, 543, 1), dtype=numpy.uint8)
        mindspore.Tensor.from_numpy(image_random)
        v_trans.encode_png(image_random, 10)

    # compression_level_type_error
    with pytest.raises(TypeError, match="Input compression_level is not of type <class 'int'>, "
                                        "but got: <class 'str'>."):
        image_random = numpy.random.randint(256, size=(876, 543, 1), dtype=numpy.uint8)
        mindspore.Tensor.from_numpy(image_random)
        v_trans.encode_png(image_random, '9')

    # compression_level_type_error
    with pytest.raises(TypeError, match="Input compression_level is not of type <class 'int'>, "
                                        "but got: <class 'float'>."):
        image_random = numpy.random.randint(256, size=(876, 543, 1), dtype=numpy.uint8)
        mindspore.Tensor.from_numpy(image_random)
        v_trans.encode_png(image_random, 9.0)


if __name__ == "__main__":
    test_encode_png_three_channels()
    test_encode_png_one_channel()
    test_encode_png_exception()
    test_encode_png_operation_01()
    test_encode_png_operation_02()
    test_encode_png_exception_01()
