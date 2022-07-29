# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
Testing ToPIL op in DE
"""
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.transforms
import mindspore.dataset.vision as vision
from mindspore import log as logger
from util import save_and_check_md5_pil

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_to_pil_01():
    """
    Feature: ToPIL op
    Description: Test ToPIL op with md5 comparison where input is already PIL image
    Expectation: Passes the md5 check test
    """
    logger.info("test_to_pil_01")

    # Generate dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        vision.Decode(True),
        # If input is already PIL image.
        vision.ToPIL(),
        vision.CenterCrop(375),
        vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.Compose(transforms)
    data1 = data1.map(operations=transform, input_columns=["image"])

    # Compare with expected md5 from images
    filename = "to_pil_01_result.npz"
    save_and_check_md5_pil(data1, filename, generate_golden=GENERATE_GOLDEN)


def test_to_pil_02():
    """
    Feature: ToPIL op
    Description: Test ToPIL op with md5 comparison where input is not a PIL image
    Expectation: Passes the md5 check test
    """
    logger.info("test_to_pil_02")

    # Generate dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    transforms = [
        # If input type is not PIL.
        vision.ToPIL(),
        vision.CenterCrop(375),
        vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.Compose(transforms)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=transform, input_columns=["image"])

    # Compare with expected md5 from images
    filename = "to_pil_02_result.npz"
    save_and_check_md5_pil(data1, filename, generate_golden=GENERATE_GOLDEN)


def test_to_pil_invalid_type():
    """
    Feature: ToPIL
    Description: Test ToPIL with invalid image type
    Expectation: Error is raised as expected
    """
    image = list(np.random.randint(0, 255, (32, 32, 3)))
    to_pil = vision.ToPIL()
    with pytest.raises(TypeError) as error_info:
        to_pil(image)
    assert "should be of type numpy.ndarray or PIL.Image.Image" in str(error_info.value)


def test_to_pil_invalid_shape():
    """
    Feature: ToPIL
    Description: Test ToPIL with invalid image shape
    Expectation: Error is raised as expected
    """
    image = np.random.randint(0, 255, (32, 32, 4, 3)).astype(np.uint8)
    to_pil = vision.ToPIL()
    with pytest.raises(ValueError) as error_info:
        to_pil(image)
    assert "dimension of input image should be 2 or 3" in str(error_info.value)

    image = np.random.randint(0, 255, (32, 32, 5)).astype(np.uint8)
    to_pil = vision.ToPIL()
    with pytest.raises(ValueError) as error_info:
        to_pil(image)
    assert "channel of input image should not exceed 4" in str(error_info.value)


def test_to_pil_invalid_dtype():
    """
    Feature: ToPIL
    Description: Test ToPIL with invalid image dtype
    Expectation: Error is raised as expected
    """
    image = np.random.randint(0, 255, (32, 32, 3)).astype(np.int16)
    to_pil = vision.ToPIL()
    with pytest.raises(TypeError) as error_info:
        to_pil(image)
    assert "image type int16 is not supported" in str(error_info.value)


if __name__ == "__main__":
    test_to_pil_01()
    test_to_pil_02()
    test_to_pil_invalid_type()
    test_to_pil_invalid_shape()
    test_to_pil_invalid_dtype()
