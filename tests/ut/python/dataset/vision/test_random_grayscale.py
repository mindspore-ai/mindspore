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
Testing RandomGrayscale op in DE
"""
import numpy as np
import os
import pytest
from PIL import Image

import mindspore.dataset.vision.transforms as vision

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
TEST_DATA_DATASET_FUNC ="../data/dataset/"

image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "png.PNG")
image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "gif.gif")














def test_random_gray_scale_exception_01():
    """
    Feature: RandomGrayscale operation
    Description: Testing the RandomGrayscale Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When the input is numpy.ndarray, the RandomGrayscale interface call fails
    image = np.random.randint(0, 255, (300, 350, 3)).astype(np.uint8)
    prob = 1.0
    random_erasing_op = vision.RandomGrayscale(prob=prob)
    with pytest.raises(AttributeError):
        random_erasing_op(image)

    # When the input is a list, the RandomGrayscale interface call fails
    with Image.open(image_jpg) as image1:
        image = np.array(image1).tolist()
        prob = 1.0
        random_erasing_op = vision.RandomGrayscale(prob=prob)
        with pytest.raises(AttributeError):
            random_erasing_op(image)

    # When parameter prob is greater than 1.0, the RandomGrayscale interface call fails
    prob = 1.1
    with pytest.raises(ValueError, match="Input prob is not within the required interval of \\[0.0, 1.0\\]."):
        vision.RandomGrayscale(prob=prob)

    # When parameter prob is negative, the RandomGrayscale interface call fails
    prob = -0.1
    with pytest.raises(ValueError, match="Input prob is not within the required interval of \\[0.0, 1.0\\]."):
        vision.RandomGrayscale(prob=prob)

    # When parameter prob is a list, the RandomGrayscale interface call fails
    prob = [0.5]
    with pytest.raises(TypeError, match="Argument prob with value \\[0.5\\] is not of "
                                        "type \\[<class 'float'>, <class 'int'>\\]."):
        vision.RandomGrayscale(prob=prob)

    # When parameter prob is a string, the RandomGrayscale interface call fails
    prob = "0.5"
    with pytest.raises(TypeError, match="Argument prob with value 0.5 is not of "
                                        "type \\[<class 'float'>, <class 'int'>\\]."):
        vision.RandomGrayscale(prob=prob)

    # When parameter prob is of bool type, the RandomGrayscale interface call fails
    prob = True
    with pytest.raises(TypeError, match="Argument prob with value True is not of "
                                        "type \\(<class 'float'>, <class 'int'>\\)."):
        vision.RandomGrayscale(prob=prob)
