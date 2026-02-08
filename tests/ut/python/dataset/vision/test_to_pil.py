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
Testing ToPIL op in DE
"""
import numpy as np
import os
import pytest

import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.transforms
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger
from util import save_and_check_md5_pil

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
TEST_DATA_DATASET_FUNC ="../data/dataset/"
DATA_DIR_1 = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")


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


def test_to_pil_operation_01():
    """
    Feature: ToPIL operation
    Description: Testing the normal functionality of the ToPIL operator
    Expectation: The Output is equal to the expected output
    """
    # ToPIL operation: After flipping, call totensor
    transforms = [vision.ToPIL(),
                  vision.RandomHorizontalFlip(0.5),
                  vision.ToTensor()]
    ds1 = ds.ImageFolderDataset(DATA_DIR_1, decode=True)
    ds1 = ds1.map(input_columns=["image"], operations=transforms)
    for _ in ds1.create_dict_iterator(output_numpy=True):
        pass

    # ToPIL operation: input equals numpy, uint8 format
    image = np.random.randint(0, 255, (32, 32, 3)).astype(np.uint8)
    topil_op = vision.ToPIL()
    out = topil_op(image)
    assert out.mode == "RGB"

    # ToPIL operation: input equals a two-dimensional numpy array
    image = np.random.randint(0, 255, (32, 32)).astype(np.uint8)
    topil_op = vision.ToPIL()
    topil_op(image)

    # ToPIL operation: input equals 4 channels
    with pytest.raises(TypeError, match="The input image type int32 is not supported when image shape is"):
        image = np.random.randint(0, 255, (32, 32, 4)).astype(np.int32)
        topil_op = vision.ToPIL()
        topil_op(image)

    # ToPIL operation: input equals 4 dimensions
    with pytest.raises(ValueError, match="The dimension of input image should be 2 or 3. Got 4."):
        image = np.random.randint(0, 255, (32, 32, 4, 3)).astype(np.uint8)
        topil_op = vision.ToPIL()
        topil_op(image)

    # ToPIL operation: input is list
    with pytest.raises(TypeError, match="The input image should be of type numpy.ndarray"
                                        " or PIL.Image.Image. Got <class 'list'>."):
        image = list(np.random.randint(0, 255, (32, 32, 3)).astype(np.uint8))
        topil_op = vision.ToPIL()
        topil_op(image)

    # ToPIL operation: input is tuple
    with pytest.raises(TypeError, match="The input image should be of type numpy.ndarray"
                                        " or PIL.Image.Image. Got <class 'tuple'>."):
        image = tuple(np.random.randint(0, 255, (200, 123, 3)).astype(np.uint8))
        topil_op = vision.ToPIL()
        topil_op(image)

    # ToPIL operation: input is ms.tensor
    with pytest.raises(TypeError, match="The input image should be of type numpy.ndarray or "
                                        "PIL.Image.Image. Got <class 'mindspore.common.tensor.Tensor'>."):
        image = ms.Tensor(np.random.randint(0, 255, (200, 123, 3)).astype(np.uint8))
        topil_op = vision.ToPIL()
        topil_op(image)

    # ToPIL operation: input is int64
    with pytest.raises(TypeError, match="The input image type int64 is not supported when image"
                                        " shape is \\[H, W, 2\\], \\[H, W, 3\\] or \\[H, W, 4\\]."):
        image = np.random.randint(0, 255, (200, 123, 3)).astype(np.int64)
        topil_op = vision.ToPIL()
        topil_op(image)

    # ToPIL operation: input is float64
    with pytest.raises(TypeError, match="The input image type float64 is not supported when image"
                                        " shape is \\[H, W, 2\\], \\[H, W, 3\\] or \\[H, W, 4\\]."):
        image = np.random.randint(0, 255, (200, 123, 3)).astype(np.float64)
        topil_op = vision.ToPIL()
        topil_op(image)

    # ToPIL operation: input is 10
    with pytest.raises(TypeError, match="The input image should be of type numpy.ndarray"
                                        " or PIL.Image.Image. Got <class 'int'>."):
        topil_op = vision.ToPIL()
        topil_op(10)

    # ToPIL operation: Passing extra parameters
    with pytest.raises(TypeError, match="takes 1 positional argument but 2 were given"):
        vision.ToPIL(True)


if __name__ == "__main__":
    test_to_pil_01()
    test_to_pil_02()
    test_to_pil_invalid_type()
    test_to_pil_invalid_shape()
    test_to_pil_invalid_dtype()
    test_to_pil_operation_01()
