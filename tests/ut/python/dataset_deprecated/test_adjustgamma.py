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
Test AdjustGamma op in Dataset
"""
import cv2
import numpy as np
import pytest
from PIL import Image
import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.vision.py_transforms as py_vision


def test_adjust_gamma_eager_invalid_image_type_c():
    """
    Feature: AdjustGamma op
    Description: Exception eager support test for AdjustGamma C++ op with error input type
    Expectation: Error input image is detected
    """

    def test_config(my_input, error_msg):
        with pytest.raises(TypeError) as error_info:
            _ = c_vision.AdjustGamma(gamma=1.2, gain=1.0)(my_input)
        assert error_msg in str(error_info.value)

    img = cv2.imread("../data/dataset/apple.jpg")
    test_config([img, img], "Input should be NumPy or PIL image, got <class 'list'>.")
    test_config((img, img), "Input should be NumPy or PIL image, got <class 'tuple'>.")

    test_config(1, "Input should be NumPy or PIL image, got <class 'int'>.")
    test_config(1.0, "Input should be NumPy or PIL image, got <class 'float'>.")
    test_config((1.0, 2.0), "Input should be NumPy or PIL image, got <class 'tuple'>.")
    test_config(((1.0, 2.0), (3.0, 4.0)), "Input should be NumPy or PIL image, got <class 'tuple'>.")
    test_config([(1.0, 2.0), (3.0, 4.0)], "Input should be NumPy or PIL image, got <class 'list'>.")


def test_adjust_gamma_eager_invalid_image_type_py():
    """
    Feature: AdjustGamma op
    Description: Exception eager support test for AdjustGamma Python op with error input type
    Expectation: Error input image is detected
    """

    def test_config(my_input, error_msg):
        with pytest.raises(TypeError) as error_info:
            _ = py_vision.AdjustGamma(gamma=1.2, gain=1.0)(my_input)
        assert error_msg in str(error_info.value)

    img = Image.open("../data/dataset/apple.jpg").convert("RGB")
    test_config([img, img], "img should be PIL image. Got <class 'list'>.")
    test_config((img, img), "img should be PIL image. Got <class 'tuple'>.")

    test_config(1, "img should be PIL image. Got <class 'int'>.")
    test_config(1.0, "img should be PIL image. Got <class 'float'>.")
    test_config((1.0, 2.0), "img should be PIL image. Got <class 'tuple'>.")
    test_config(((1.0, 2.0), (3.0, 4.0)), "img should be PIL image. Got <class 'tuple'>.")
    test_config([(1.0, 2.0), (3.0, 4.0)], "img should be PIL image. Got <class 'list'>.")


def test_adjust_gamma_eager_image_type():
    """
    Feature: AdjustGamma op
    Description: Test AdjustGamma op eager support test for variety of image input types
    Expectation: Receive non-None output image from op
    """

    def test_config(my_input):
        my_output = c_vision.AdjustGamma(gamma=1.2, gain=1.0)(my_input)
        assert my_output is not None

    # Test with OpenCV images
    img = cv2.imread("../data/dataset/apple.jpg")
    test_config(img)

    # Test with NumPy array input
    img = np.random.randint(0, 1, (100, 100, 3)).astype(np.uint8)
    test_config(img)

    # Test with PIL Image
    img = Image.open("../data/dataset/apple.jpg").convert("RGB")
    test_config(img)


if __name__ == '__main__':
    test_adjust_gamma_eager_invalid_image_type_c()
    test_adjust_gamma_eager_invalid_image_type_py()
    test_adjust_gamma_eager_image_type()
