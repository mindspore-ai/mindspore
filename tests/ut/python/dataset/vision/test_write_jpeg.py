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
Testing write_jpeg
"""
import os
import cv2
import numpy
import pytest

from mindspore import Tensor
from mindspore.dataset import vision


def test_write_jpeg_three_channels():
    """
    Feature: write_jpeg
    Description: Write the image containing three channels into a JPEG file
    Expectation: The file should be written and removed
    """

    def write_jpeg_three_channels(filename_param, image_param, quality_param=75):
        """
        a function used for writing with three channels image
        """
        vision.write_jpeg(filename_param, image_param, quality_param)
        image_2_numpy = cv2.imread(filename_param, cv2.IMREAD_UNCHANGED)
        os.remove(filename_param)
        assert image_2_numpy.shape == (2268, 4032, 3)

    filename_1 = "../data/dataset/apple.jpg"
    mode = cv2.IMREAD_UNCHANGED
    image_bgr = cv2.imread(filename_1, mode)
    image_1_numpy = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_1_tensor = Tensor.from_numpy(image_1_numpy)
    filename_2 = filename_1 + ".test_write_jpeg_three_channels.jpg"

    # Test writing numpy.ndarray
    write_jpeg_three_channels(filename_2, image_1_numpy)

    # Test writing Tensor and quality 1, 75, 100
    for quality in (1, 75, 100):
        write_jpeg_three_channels(filename_2, image_1_tensor, quality)

    # Test with three channels 2268*4032*3 random uint8, the quality is 50
    image_random = numpy.ndarray(shape=(2268, 4032, 3), dtype=numpy.uint8)
    write_jpeg_three_channels(filename_2, image_random, 50)


def test_write_jpeg_one_channel():
    """
    Feature: write_jpeg
    Description: Write the grayscale image into a JPEG file
    Expectation: The file should be written and removed
    """

    def write_jpeg_one_channel(filename_param, image_param, quality_param=75):
        """
        a function used for writing with three channels image
        """
        vision.write_jpeg(filename_param, image_param, quality_param)
        image_2_numpy = cv2.imread(filename_param, cv2.IMREAD_UNCHANGED)
        os.remove(filename_param)
        assert image_2_numpy.shape == (2268, 4032)

    filename_1 = "../data/dataset/apple.jpg"
    mode = cv2.IMREAD_UNCHANGED
    image_1_numpy = cv2.imread(filename_1, mode)
    filename_2 = filename_1 + ".test_write_jpeg_one_channel.jpg"
    image_grayscale = cv2.cvtColor(image_1_numpy, cv2.COLOR_BGR2GRAY)
    image_grayscale_tensor = Tensor.from_numpy(image_grayscale)

    # Test writing numpy.ndarray
    write_jpeg_one_channel(filename_2, image_grayscale)

    # Test writing Tensor and quality 1, 75, 100
    for quality in (1, 75, 100):
        write_jpeg_one_channel(filename_2, image_grayscale_tensor, quality)

    # Test with three channels 2268*4032 random uint8
    image_random = numpy.ndarray(shape=(2268, 4032), dtype=numpy.uint8)
    write_jpeg_one_channel(filename_2, image_random)

    # Test with one channels 2268*4032*1 random uint8, the quality is 50
    image_random = numpy.ndarray(shape=(2268, 4032, 1), dtype=numpy.uint8)
    write_jpeg_one_channel(filename_2, image_random, 50)


def test_write_jpeg_exception():
    """
    Feature: write_jpeg
    Description: Test write_jpeg with an invalid parameter
    Expectation: Error is caught when the parameter is invalid
    """

    def test_invalid_param(filename_param, image_param, quality_param, error, error_msg):
        """
        a function used for checking correct error and message with invalid parameter
        """
        with pytest.raises(error) as error_info:
            vision.write_jpeg(filename_param, image_param, quality_param)
        assert error_msg in str(error_info.value)

    filename_1 = "../data/dataset/apple.jpg"
    mode = cv2.IMREAD_UNCHANGED
    image_1_numpy = cv2.imread(filename_1, mode)
    image_1_tensor = Tensor.from_numpy(image_1_numpy)

    # Test with a directory name
    wrong_filename = "../data/dataset/"
    error_message = "Invalid file path, " + wrong_filename + " is not a regular file."
    test_invalid_param(wrong_filename, image_1_numpy, 75, RuntimeError, error_message)

    # Test with an invalid filename
    wrong_filename = "/dev/cdrom/0"
    error_message = "No such file or directory"
    test_invalid_param(wrong_filename, image_1_tensor, 75, RuntimeError, error_message)

    # Test with an invalid type for the filename
    error_message = "Input filename is not of type"
    test_invalid_param(0, image_1_numpy, 75, TypeError, error_message)

    # Test with an invalid type for the data
    filename_2 = filename_1 + ".test_write_jpeg.jpg"
    error_message = "Input image is not of type"
    test_invalid_param(filename_2, 0, 75, TypeError, error_message)

    # Test with invalid float elements
    invalid_data = numpy.ndarray(shape=(10, 10), dtype=float)
    error_message = "The type of the elements of image should be UINT8"
    test_invalid_param(filename_2, invalid_data, 75, RuntimeError, error_message)

    # Test with invalid image with only one dimension
    invalid_data = numpy.ndarray(shape=(10), dtype=numpy.uint8)
    error_message = "The image has invalid dimensions"
    test_invalid_param(filename_2, invalid_data, 75, RuntimeError, error_message)

    # Test with invalid image with four dimensions
    invalid_data = numpy.ndarray(shape=(1, 2, 3, 4), dtype=numpy.uint8)
    test_invalid_param(filename_2, invalid_data, 75, RuntimeError, error_message)

    # Test with invalid image with two channels
    invalid_data = numpy.ndarray(shape=(2, 3, 2), dtype=numpy.uint8)
    error_message = "The image has invalid channels"
    test_invalid_param(filename_2, invalid_data, 75, RuntimeError, error_message)

    # Test with invalid quality
    invalid_data = numpy.ndarray(shape=(2, 3, 2), dtype=numpy.uint8)
    error_message = "The image has invalid channels"
    test_invalid_param(filename_2, invalid_data, 75, RuntimeError, error_message)

    # Test with an invalid integer for the quality 0, 101
    error_message = "Invalid quality"
    test_invalid_param(filename_2, image_1_numpy, 0, RuntimeError, error_message)
    test_invalid_param(filename_2, image_1_numpy, 101, RuntimeError, error_message)

    # Test with an invalid type for the quality
    error_message = "Input quality is not of type"
    test_invalid_param(filename_2, image_1_numpy, 75.0, TypeError, error_message)


def test_write_jpeg_operation_01():
    """
    Feature: write_jpeg operation
    Description: Testing the normal functionality of the write_jpeg operator
    Expectation: The Output is equal to the expected output
    """
    # Normal scenario using write_jpeg
    data_jpeg = numpy.random.randint(0, 255, (128, 256, 3)).astype(numpy.uint8)
    save_file_name = 'test_write_jpeg.jpg'
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)
    vision.write_jpeg(filename=save_path, image=data_jpeg, quality=75)
    if os.path.exists(save_path):
        assert True
        os.remove(save_path)
    else:
        assert False, "No JPEG file generated"

    # write_jpeg: does not pass the quality parameter
    data_jpeg = numpy.random.randint(0, 255, (128, 256, 3)).astype(numpy.uint8)
    save_file_name = 'test_write_jpeg.jpg'
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)

    vision.write_jpeg(filename=save_path, image=data_jpeg)
    if os.path.exists(save_path):
        assert True
        os.remove(save_path)
    else:
        assert False, "No JPEG file generated"

    # write_jpeg: positional argument passing
    data_jpeg = numpy.random.randint(0, 255, (128, 256, 3)).astype(numpy.uint8)
    save_file_name = 'test_write_jpeg.jpg'
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)

    vision.write_jpeg(save_path, data_jpeg, 6)
    if os.path.exists(save_path):
        assert True
        os.remove(save_path)
    else:
        assert False, "No JPEG file generated"

    # write_jpeg:: Image parameter passed as a Tensor
    data_jpeg = numpy.random.randint(0, 255, (128, 256, 3)).astype(numpy.uint8)
    data_tensor = Tensor(data_jpeg)
    save_file_name = 'test_write_jpeg.jpg'
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)
    vision.write_jpeg(filename=save_path, image=data_tensor)
    if os.path.exists(save_path):
        assert True
        os.remove(save_path)
    else:
        assert False, "No JPEG file generated"

    # write_jpeg: Image input two-dimensional data
    data_jpeg = numpy.random.randint(0, 255, (128, 3)).astype(numpy.uint8)
    save_file_name = 'test_write_jpeg.jpg'
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)
    vision.write_jpeg(filename=save_path, image=data_jpeg)
    if os.path.exists(save_path):
        assert True
        os.remove(save_path)
    else:
        assert False, "No JPEG file generated"

    # write_jpeg: Input 3D single-channel data
    data_jpeg = numpy.random.randint(0, 255, (128, 256, 1)).astype(numpy.uint8)
    save_file_name = 'test_write_jpeg.jpg'
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)
    vision.write_jpeg(filename=save_path, image=data_jpeg)
    if os.path.exists(save_path):
        assert True
        os.remove(save_path)
    else:
        assert False, "No JPEG file generated"

    # write_jpeg: Verify the quality parameter within the range of 1 to 100.
    for quality in [1, 25, 75, 100]:
        data_jpeg = numpy.random.randint(0, 255, (128, 256, 3)).astype(numpy.uint8)
        save_file_name = 'test_write_jpeg.jpg'
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)
        vision.write_jpeg(filename=save_path, image=data_jpeg, quality=quality)
        if os.path.exists(save_path):
            assert True
            os.remove(save_path)
        else:
            assert False, "No JPEG file generated"


def test_write_jpeg_exception_01():
    """
    Feature: write_jpeg operation
    Description: Testing the write_jpeg Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # write_jpeg: No parameters are passed.
    with pytest.raises(TypeError,
                       match=r"write_jpeg\(\) missing 2 required positional arguments: 'filename' and 'image'"):
        vision.write_jpeg()

    # write_jpeg: Do not pass the filename parameter
    data_jpeg = numpy.random.randint(0, 255, (128, 256, 3)).astype(numpy.uint8)
    with pytest.raises(TypeError,
                       match=r"write_jpeg\(\) missing 1 required positional argument: 'filename'"):
        vision.write_jpeg(image=data_jpeg, quality=75)

    # write_jpeg: Do not pass the image parameter
    save_file_name = 'test_write_jpeg.jpg'
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)
    with pytest.raises(TypeError,
                       match=r"write_jpeg\(\) missing 1 required positional argument: 'image'"):
        vision.write_jpeg(filename=save_path, quality=75)

    # write_jpeg: Pass an additional keyword argument
    data_jpeg = numpy.random.randint(0, 255, (128, 256, 3)).astype(numpy.uint8)
    save_file_name = 'test_write_jpeg.jpg'
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)
    with pytest.raises(TypeError,
                       match=r"write_jpeg\(\) got an unexpected keyword argument 'sup'"):
        vision.write_jpeg(filename=save_path, image=data_jpeg, quality=75, sup=4)

    # write_jpeg: One more positional parameter
    data_jpeg = numpy.random.randint(0, 255, (128, 256, 3)).astype(numpy.uint8)
    save_file_name = 'test_write_jpeg.jpg'
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)
    with pytest.raises(TypeError,
                       match=r"write_jpeg\(\) takes from 2 to 3 positional arguments but 4 were given"):
        vision.write_jpeg(save_path, data_jpeg, 6, 4)

    # write_jpeg: filename parameter is not a string type validation
    for filenames in [1, 1.1, ['./jpeg.jpg'], {'a': 1}, None, False]:
        data_jpeg = numpy.random.randint(0, 255, (128, 256, 3)).astype(numpy.uint8)
        with pytest.raises(TypeError, match=r"Input filename is not of type <class 'str'>"):
            vision.write_jpeg(filenames, data_jpeg)

    # write_jpeg: Validation of filename parameter path correctness
    data_jpeg = numpy.random.randint(0, 255, (128, 256, 3)).astype(numpy.uint8)
    with pytest.raises(RuntimeError, match=r"No such file or directory"):
        vision.write_jpeg('./aaa/bb', data_jpeg)

    # write_jpeg: General Type Validation for the image Parameter
    for images in [1, 1.1, ['./jpeg.jpg'], {'a': 1}, None, False, 'str']:
        save_file_name = 'test_write_jpeg.jpg'
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)

        with pytest.raises(TypeError,
                           match=r"Input image is not of type <class 'numpy.ndarray'> "
                                 r"or <class 'mindspore.common.tensor.Tensor'>"):
            vision.write_jpeg(filename=save_path, image=images)

    # write_jpeg: Image parameter numpy type validation
    for dtypes in [numpy.uint16, numpy.uint32, numpy.uint64, numpy.float16, numpy.float32, numpy.float64, numpy.str_]:
        data_jpeg = numpy.random.randint(0, 255, (128, 256, 3)).astype(dtypes)
        save_file_name = 'test_write_jpeg.jpg'
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)
        with pytest.raises(RuntimeError, match=r"WriteJpeg: The type of the elements of image should be UINT8"):
            vision.write_jpeg(filename=save_path, image=data_jpeg)

    # write_jpeg: Input four-dimensional data
    data_jpeg = numpy.random.randint(0, 255, (128, 256, 3, 3)).astype(numpy.uint8)
    save_file_name = 'test_write_jpeg.jpg'
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)
    with pytest.raises(RuntimeError,
                       match=r"WriteJpeg: The image has invalid dimensions. It should have two or three dimensions"):
        vision.write_jpeg(filename=save_path, image=data_jpeg)

    # write_jpeg: Input one-dimensional data
    data_jpeg = numpy.random.randint(0, 255, (3,)).astype(numpy.uint8)
    save_file_name = 'test_write_jpeg.jpg'
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)

    with pytest.raises(RuntimeError,
                       match=r"WriteJpeg: The image has invalid dimensions. It should have two or three dimensions"):
        vision.write_jpeg(filename=save_path, image=data_jpeg)

    # write_jpeg: Input 3D dual-channel data
    data_jpeg = numpy.random.randint(0, 255, (128, 256, 2)).astype(numpy.uint8)
    save_file_name = 'test_write_jpeg.jpg'
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)
    with pytest.raises(RuntimeError, match=r"The image has invalid channels. It should have 1 or 3 channels"):
        vision.write_jpeg(filename=save_path, image=data_jpeg)

    # write_jpeg: Input three-dimensional four-channel data
    data_jpeg = numpy.random.randint(0, 255, (128, 256, 4)).astype(numpy.uint8)
    save_file_name = 'test_write_jpeg.jpg'
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)
    with pytest.raises(RuntimeError, match=r"The image has invalid channels. It should have 1 or 3 channels"):
        vision.write_jpeg(filename=save_path, image=data_jpeg)


def test_write_jpeg_exception_02():
    """
    Feature: write_jpeg operation
    Description: Testing the write_jpeg Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # write_jpeg: quality is not an int type
    for quality in [1.1, 'a', (12, 4), {'a': 1}, None, [1, 2, 3]]:
        data_jpeg = numpy.random.randint(0, 255, (128, 256, 3)).astype(numpy.uint8)
        save_file_name = 'test_write_jpeg.jpg'
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)
        with pytest.raises(TypeError, match=r"Input quality is not of type <class 'int'>"):
            vision.write_jpeg(filename=save_path, image=data_jpeg, quality=quality)

    # write_jpeg: Quality parameter out of range 1-100
    for quality in [0, 101]:
        data_jpeg = numpy.random.randint(0, 255, (128, 256, 3)).astype(numpy.uint8)
        save_file_name = 'test_write_jpeg.jpg'
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_file_name)

        with pytest.raises(RuntimeError, match=r"should be in range of \[1, 100\]"):
            vision.write_jpeg(filename=save_path, image=data_jpeg, quality=quality)


if __name__ == "__main__":
    test_write_jpeg_three_channels()
    test_write_jpeg_one_channel()
    test_write_jpeg_exception()
    test_write_jpeg_operation_01()
    test_write_jpeg_exception_01()
    test_write_jpeg_exception_02()
