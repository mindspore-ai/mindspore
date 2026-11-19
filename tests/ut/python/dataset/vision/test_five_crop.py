# Copyright 2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Testing FiveCrop in DE
"""
import numpy as np
import os
import pytest
from PIL import Image

import mindspore.dataset.vision.transforms as vision

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
IMAGENET_DIR = "../data/dataset/testPK/data"
TEST_DATA_DATASET_FUNC ="../data/dataset/"

GENERATE_GOLDEN = False














def test_five_crop_exception_01():
    """
    Feature: FiveCrop operation
    Description: Testing the FiveCrop Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # FiveCrop operator: Test no size
    with pytest.raises(TypeError, match="missing a required argument: 'size'"):
        vision.FiveCrop()

    # FiveCrop operator: Test size is 'wajda^% 12^*l'
    size = "wajda^% 12^*l"
    with pytest.raises(TypeError, match=(r"Argument size with value wajda\^\% 12\^\*l is not of "
                                         r"type \[<class 'int'>, <class 'list'>, <class 'tuple'>\]")):
        vision.FiveCrop(size=size)

    # FiveCrop operator: Test size is 0
    size = 0
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[1, 16777216\\]."):
        vision.FiveCrop(size=size)

    # FiveCrop operator: Test image is numpy array
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image_data:
        image = np.array(image_data)
        size = [100, 100]
        five_crop_op = vision.FiveCrop(size=size)
        with pytest.raises(TypeError, match="img should be PIL image. Got <class 'numpy.ndarray'>."):
            five_crop_op(image)

    # FiveCrop operator: Test image is numpy list
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image_data:
        image = np.array(image_data).tolist()
        size = [100, 100]
        five_crop_op = vision.FiveCrop(size=size)
        with pytest.raises(TypeError, match="img should be PIL image. Got <class 'list'>."):
            five_crop_op(image)

    # FiveCrop operator: Test no image
    size = [100, 100]
    five_crop_op = vision.FiveCrop(size=size)
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'img'"):
        five_crop_op()

    # Test size > image.shape
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")
    with Image.open(image_jpg) as image:
        size = [433, 576]
        five_crop_op = vision.FiveCrop(size=size)
        with pytest.raises(ValueError,
                           match="Crop size \\[433, 576\\] is larger than input image size \\(432, 576\\)."):
            five_crop_op(image)

    # FiveCrop operator: Test size is float
    size = 50.0
    with pytest.raises(TypeError, match="Argument size with value 50.0 is not of "
                                        "type \\[<class 'int'>, <class 'list'>, <class 'tuple'>\\]."):
        vision.FiveCrop(size=size)

    # FiveCrop operator: Test size is 1-tuple
    size = (50, 60, 70)
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple \\(h, w\\) of length 2."):
        vision.FiveCrop(size=size)

    # FiveCrop operator: Test size is list
    size = [60]
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple \\(h, w\\) of length 2."):
        vision.FiveCrop(size=size)

    # FiveCrop operator: Test size is numpy
    size = np.array([10, 20])
    with pytest.raises(TypeError, match="Argument size with value \\[10 20\\] is not of "
                                        "type \\[<class 'int'>, <class 'list'>, <class 'tuple'>\\]."):
        vision.FiveCrop(size=size)
