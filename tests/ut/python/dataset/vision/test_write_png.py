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
Testing write_png
"""
import os
import cv2
import numpy
import pytest

from mindspore import Tensor
from mindspore.dataset.transforms import vision


def test_write_png_three_channels():
    """
    Feature: Test write_png
    Description: Write the image containing three channels into a PNG file
    Expectation: The file should be written and removed
    """

    def write_png_three_channels(filename_param, image_param, compression_level_param=6):
        """
        a function used for writing with three channels image
        """
        vision.write_png(filename_param, image_param, compression_level_param)
        image_2_numpy = cv2.imread(filename_param, cv2.IMREAD_UNCHANGED)
        os.remove(filename_param)
        assert image_2_numpy.shape == (2268, 4032, 3)

    filename_1 = "../data/dataset/apple.jpg"
    mode = cv2.IMREAD_UNCHANGED
    image_bgr = cv2.imread(filename_1, mode)
    image_1_numpy = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_1_tensor = Tensor.from_numpy(image_1_numpy)
    filename_2 = filename_1 + ".test_write_png_three_channels.png"

    # Test writing numpy.ndarray
    write_png_three_channels(filename_2, image_1_numpy)

    # Test writing Tensor and compression_level 0, 6, 9
    for compression_level in (0, 6, 9):
        write_png_three_channels(filename_2, image_1_tensor, compression_level)

    # Test with three channels 2268*4032*3 random uint8, the compression_level is 5
    image_random = numpy.ndarray(shape=(2268, 4032, 3), dtype=numpy.uint8)
    write_png_three_channels(filename_2, image_random, 5)


def test_write_png_one_channel():
    """
    Feature: Test write_png
    Description: Write the grayscale image into a PNG file
    Expectation: The file should be written and removed
    """

    def write_png_one_channel(filename_param, image_param, compression_level_param=6):
        """
        a function used for writing grayscale image
        """
        vision.write_png(filename_param, image_param, compression_level_param)
        image_2_numpy = cv2.imread(filename_param, cv2.IMREAD_UNCHANGED)
        os.remove(filename_param)
        assert image_2_numpy.shape == (2268, 4032)

    filename_1 = "../data/dataset/apple.jpg"
    mode = cv2.IMREAD_UNCHANGED
    image_1_numpy = cv2.imread(filename_1, mode)
    filename_2 = filename_1 + ".test_write_png_one_channel.png"
    image_grayscale = cv2.cvtColor(image_1_numpy, cv2.COLOR_BGR2GRAY)
    image_grayscale_tensor = Tensor.from_numpy(image_grayscale)

    # Test writing numpy.ndarray
    write_png_one_channel(filename_2, image_grayscale)

    # Test writing Tensor and compression_level 0, 6, 9
    for compression_level in (0, 6, 9):
        write_png_one_channel(filename_2, image_grayscale_tensor, compression_level)

    # Test with one channel 2268*4032 random uint8
    image_random = numpy.ndarray(shape=(2268, 4032), dtype=numpy.uint8)
    write_png_one_channel(filename_2, image_random)

    # Test with one channel 12268*4032*1 random uint8, the compression_level is 5
    image_random = numpy.ndarray(shape=(2268, 4032, 1), dtype=numpy.uint8)
    write_png_one_channel(filename_2, image_random, 5)


def test_write_png_exception():
    """
    Feature: Test write_png
    Description: Test write_png op with invalid parameter
    Expectation: Error is caught when the parameter is invalid
    """

    def test_invalid_param(filename_param, image_param, compression_level_param, error, error_msg):
        """
        a function used for checking correct error and message with invalid parameter
        """
        with pytest.raises(error) as error_info:
            vision.write_png(filename_param, image_param, compression_level_param)
        assert error_msg in str(error_info.value)

    filename_1 = "../data/dataset/apple.jpg"
    mode = cv2.IMREAD_UNCHANGED
    image_1_numpy = cv2.imread(filename_1, mode)
    image_1_tensor = Tensor.from_numpy(image_1_numpy)
    filename_2 = filename_1 + ".test_write_png.png"

    # Test with a directory name
    wrong_filename = "../data/dataset/"
    error_message = "Invalid file path, " + wrong_filename + " is not a regular file."
    test_invalid_param(wrong_filename, image_1_numpy, 6, RuntimeError, error_message)

    # Test with an invalid filename
    wrong_filename = "/dev/cdrom/0"
    error_message = "No such file or directory"
    test_invalid_param(wrong_filename, image_1_tensor, 6, RuntimeError, error_message)

    # Test with an invalid type for the filename
    error_message = "Input filename is not of type"
    test_invalid_param(0, image_1_numpy, 6, TypeError, error_message)

    # Test with an invalid type for the data
    error_message = "The input image is not of type"
    test_invalid_param(filename_2, 0, 6, TypeError, error_message)

    # Test with invalid float elements
    invalid_data = numpy.ndarray(shape=(10, 10), dtype=float)
    error_message = "The type of the elements of image should be UINT8"
    test_invalid_param(filename_2, invalid_data, 6, RuntimeError, error_message)

    # Test with invalid image with only one dimension
    invalid_data = numpy.ndarray(shape=(10), dtype=numpy.uint8)
    error_message = "The image has invalid dimensions"
    test_invalid_param(filename_2, invalid_data, 6, RuntimeError, error_message)

    # Test with invalid image with four dimensions
    invalid_data = numpy.ndarray(shape=(1, 2, 3, 4), dtype=numpy.uint8)
    error_message = "The image has invalid dimensions"
    test_invalid_param(filename_2, invalid_data, 6, RuntimeError, error_message)

    # Test with invalid image with two channels
    invalid_data = numpy.ndarray(shape=(2, 3, 2), dtype=numpy.uint8)
    error_message = "The image has invalid channels"
    test_invalid_param(filename_2, invalid_data, 6, RuntimeError, error_message)

    # Test writing with invalid compression_level -1, 10
    error_message = "Invalid compression_level"
    test_invalid_param(filename_2, image_1_numpy, -1, RuntimeError, error_message)
    test_invalid_param(filename_2, image_1_tensor, 10, RuntimeError, error_message)

    # Test writing with the invalid compression_level 6.5
    error_message = "Input compression_level is not of type"
    test_invalid_param(filename_2, image_1_tensor, 6.5, TypeError, error_message)


def test_write_png_operation_01():
    """
    Feature: write_png operation
    Description: Testing the normal functionality of the write_png operator
    Expectation: The Output is equal to the expected output
    """
    # Use write_png in normal scenarios
    data_png = numpy.random.randint(0, 255, (128, 256, 3)).astype(numpy.uint8)
    save_file_name = 'test_write_png.png'
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)
    vision.write_png(filename=save_path, image=data_png, compression_level=6)
    if os.path.exists(save_path):
        assert True
        os.remove(save_path)
    else:
        assert False, "No PNG file generated"

    # write_png does not pass the compression_level parameter
    data_png = numpy.random.randint(0, 255, (128, 256, 3)).astype(numpy.uint8)
    save_file_name = 'test_write_png.png'
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)

    vision.write_png(filename=save_path, image=data_png)
    if os.path.exists(save_path):
        assert True
        os.remove(save_path)
    else:
        assert False, "No PNG file generated"

    # write_png positional parameter passing
    data_png = numpy.random.randint(0, 255, (128, 256, 3)).astype(numpy.uint8)
    save_file_name = 'test_write_png.png'
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)

    vision.write_png(save_path, data_png, 6)
    if os.path.exists(save_path):
        assert True
        os.remove(save_path)
    else:
        assert False, "No PNG file generated"

    # The image parameter passed to write_png is a Tensor.
    data_png = numpy.random.randint(0, 255, (128, 256, 3)).astype(numpy.uint8)
    data_tensor = Tensor(data_png)
    save_file_name = 'test_write_png.png'
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)
    vision.write_png(filename=save_path, image=data_tensor)
    if os.path.exists(save_path):
        assert True
        os.remove(save_path)
    else:
        assert False, "No PNG file generated"

    # The image parameter of write_png accepts two-dimensional data.
    data_png = numpy.random.randint(0, 255, (128, 3)).astype(numpy.uint8)
    save_file_name = 'test_write_png.png'
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)
    vision.write_png(filename=save_path, image=data_png)
    if os.path.exists(save_path):
        assert True
        os.remove(save_path)
    else:
        assert False, "No PNG file generated"

    # The image parameter of write_png accepts three-dimensional single-channel data.
    data_png = numpy.random.randint(0, 255, (128, 256, 1)).astype(numpy.uint8)
    save_file_name = 'test_write_png.png'
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)
    vision.write_png(filename=save_path, image=data_png)
    if os.path.exists(save_path):
        assert True
        os.remove(save_path)
    else:
        assert False, "No PNG file generated"

    # write_png: Tested within compression_level range 0-9
    for compression_level in [0, 3, 6, 9]:
        data_png = numpy.random.randint(0, 255, (128, 256, 3)).astype(numpy.uint8)
        save_file_name = 'test_write_png.png'
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)
        vision.write_png(filename=save_path, image=data_png, compression_level=compression_level)
        if os.path.exists(save_path):
            assert True
            os.remove(save_path)
        else:
            assert False, "No PNG file generated"


def test_write_png_exception_01():
    """
    Feature: write_png operation
    Description: Testing the write_png Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # write_png does not pass any parameters
    error_message = "write_png() missing 2 required positional arguments: 'filename' and 'image'"
    with pytest.raises(TypeError) as error_info:
        vision.write_png()
    assert error_message in str(error_info.value)

    # write_png does not pass the filename parameter
    data_png = numpy.random.randint(0, 255, (128, 256, 3)).astype(numpy.uint8)
    with pytest.raises(TypeError, match=r"write_png\(\) missing 1 required positional argument: 'filename'"):
        vision.write_png(image=data_png, compression_level=6)

    # write_png does not pass the image parameter
    save_file_name = 'test_write_png.png'
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)
    with pytest.raises(TypeError, match=r"write_png\(\) missing 1 required positional argument: 'image'"):
        vision.write_png(filename=save_path, compression_level=6)

    # write_png: Pass an additional keyword argument
    data_png = numpy.random.randint(0, 255, (128, 256, 3)).astype(numpy.uint8)
    save_file_name = 'test_write_png.png'
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)
    with pytest.raises(TypeError, match=r"write_png\(\) got an unexpected keyword argument 'sup'"):
        vision.write_png(filename=save_path, image=data_png, compression_level=6, sup=4)

    # write_png has an additional positional argument
    data_png = numpy.random.randint(0, 255, (128, 256, 3)).astype(numpy.uint8)
    save_file_name = 'test_write_png.png'
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)

    with pytest.raises(TypeError, match=r"write_png\(\) takes from 2 to 3 positional arguments but 4 were given"):
        vision.write_png(save_path, data_png, 6, 4)

    # write_png filename parameter is not a string type validation
    data_png = numpy.random.randint(0, 255, (128, 256, 3)).astype(numpy.uint8)
    with pytest.raises(TypeError, match=r"Input filename is not of type <class 'str'>"):
        vision.write_png([1, 2, 3], data_png)

    # Validity Check for the Path of the filename Parameter in write_png
    data_png = numpy.random.randint(0, 255, (128, 256, 3)).astype(numpy.uint8)
    error_message = "No such file or directory"
    with pytest.raises(RuntimeError) as error_info:
        vision.write_png('./aaa/bb', data_png)
    assert error_message in str(error_info.value)

    # Type validation for the image parameter in write_png
    save_file_name = 'test_write_png.png'
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)
    with pytest.raises(TypeError,
                       match=r"The input image is not of type <class 'numpy.ndarray'> "
                             r"or <class 'mindspore.common.tensor.Tensor'>"):
        vision.write_png(filename=save_path, image='str')

    # Validate numpy type for image parameter in write_png
    for dtypes in [numpy.uint16, numpy.uint32, numpy.uint64, numpy.float16, numpy.float32, numpy.float64, numpy.str_]:
        data_png = numpy.random.randint(0, 255, (128, 256, 3)).astype(dtypes)
        save_file_name = 'test_write_png.png'
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)
        with pytest.raises(RuntimeError, match=r"WritePng: The type of the elements of image should be UINT8"):
            vision.write_png(filename=save_path, image=data_png)

    # The image parameter for write_png accepts four-dimensional data.
    data_png = numpy.random.randint(0, 255, (128, 256, 3, 3)).astype(numpy.uint8)
    save_file_name = 'test_write_png.png'
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)
    with pytest.raises(RuntimeError,
                       match=r"WritePng: The image has invalid dimensions. It should have two or three dimensions"):
        vision.write_png(filename=save_path, image=data_png)

    # The image parameter for write_png accepts one-dimensional data.
    data_png = numpy.random.randint(0, 255, (3,)).astype(numpy.uint8)
    save_file_name = 'test_write_png.png'
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)
    with pytest.raises(RuntimeError,
                       match=r"WritePng: The image has invalid dimensions. It should have two or three dimensions"):
        vision.write_png(filename=save_path, image=data_png)

    # write_png: Image input with three-dimensional dual-channel data
    data_png = numpy.random.randint(0, 255, (128, 256, 2)).astype(numpy.uint8)
    save_file_name = 'test_write_png.png'
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)
    with pytest.raises(RuntimeError, match=r"The image has invalid channels. It should have 1 or 3 channels"):
        vision.write_png(filename=save_path, image=data_png)

    # write_png: Image input with three-dimensional four-channel data
    data_png = numpy.random.randint(0, 255, (128, 256, 4)).astype(numpy.uint8)
    save_file_name = 'test_write_png.png'
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)

    with pytest.raises(RuntimeError, match=r"The image has invalid channels. It should have 1 or 3 channels"):
        vision.write_png(filename=save_path, image=data_png)

    # write_png: compression_level parameter type test
    for compression_level in [1.1, 'a', (12, 4), {'a': 1}, None, [1, 2, 3]]:
        data_png = numpy.random.randint(0, 255, (128, 256, 3)).astype(numpy.uint8)
        save_file_name = 'test_write_png.png'
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)
        with pytest.raises(TypeError, match=r"Input compression_level is not of type <class 'int'>"):
            vision.write_png(filename=save_path, image=data_png, compression_level=compression_level)


def test_write_png_exception_02():
    """
    Feature: write_png operation
    Description: Testing the write_png Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # write_png: compression_level parameter range test
    data_png = numpy.random.randint(0, 255, (128, 256, 3)).astype(numpy.uint8)
    save_file_name = 'test_write_png.png'
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)
    with pytest.raises(RuntimeError, match=r"WritePng: Invalid compression_level -1, should be in range of \[0, 9\]."):
        vision.write_png(filename=save_path, image=data_png, compression_level=-1)

    # write_png: compression_level parameter range test
    data_png = numpy.random.randint(0, 255, (128, 256, 3)).astype(numpy.uint8)
    save_file_name = 'test_write_png.png'
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)
    with pytest.raises(RuntimeError, match=r"WritePng: Invalid compression_level 10, should be in range of \[0, 9\]."):
        vision.write_png(filename=save_path, image=data_png, compression_level=10)


if __name__ == "__main__":
    test_write_png_three_channels()
    test_write_png_one_channel()
    test_write_png_exception()
    test_write_png_operation_01()
    test_write_png_exception_01()
    test_write_png_exception_02()
