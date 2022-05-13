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
Testing ToType op in DE
"""
import numpy as np
import pytest
import mindspore.dataset as ds
import mindspore.dataset.transforms.py_transforms
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore import log as logger
from ..dataset.util import save_and_check_md5

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_to_type_op():
    """
    Feature: ToType op
    Description: Test ToType op with numpy.dtype output_type arg.
    Expectation: Data results are correct and the same
    """
    logger.info("test_to_type_op")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms1 = [
        py_vision.Decode(),
        py_vision.ToTensor(),
        # Note: Convert the datatype from float32 to int16
        py_vision.ToType(output_type=np.int16)
    ]
    transform1 = mindspore.dataset.transforms.py_transforms.Compose(transforms1)
    data1 = data1.map(operations=transform1, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms2 = [
        py_vision.Decode(),
        py_vision.ToTensor()
    ]
    transform2 = mindspore.dataset.transforms.py_transforms.Compose(transforms2)
    data2 = data2.map(operations=transform2, input_columns=["image"])

    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image1 = item1["image"]
        image2 = item2["image"]

        assert isinstance(image1, np.ndarray)
        assert isinstance(image2, np.ndarray)
        assert image1.dtype == np.int16
        assert image2.dtype == np.float32
        assert image1.shape == image2.shape


def test_to_type_01():
    """
    Feature: ToType op
    Description: Test ToType op with valid numpy.dtype input
    Expectation: Dataset pipeline runs successfully and md5 results are verified
    """
    logger.info("test_to_type_01")

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        py_vision.Decode(),
        py_vision.ToTensor(),
        # Note: Convert the datatype from float32 to int32
        py_vision.ToType(np.int32)
    ]
    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])

    # Compare with expected md5 from images
    filename = "to_type_01_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)


def test_to_type_02():
    """
    Feature: ToType op
    Description: Test ToType op with valid string input
    Expectation: Dataset pipeline runs successfully and md5 results are verified
    """

    logger.info("test_to_type_02")
    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        py_vision.Decode(),
        py_vision.ToTensor(),
        # Note: Convert to type int
        py_vision.ToType('int')
    ]
    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])

    # Compare with expected md5 from images
    filename = "to_type_02_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)


def test_to_type_03():
    """
    Feature: ToType op
    Description: Test ToType op with invalid input image type
    Expectation: Invalid input image type is detected.
    """

    logger.info("test_to_type_03")

    with pytest.raises(RuntimeError) as error_info:
        # Generate dataset
        data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
        transforms = [
            py_vision.Decode(),
            # Note: If the object is not numpy, e.g. PIL image, RunTimeError will raise
            py_vision.ToType(np.int32)
        ]
        transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
        data = data.map(operations=transform, input_columns=["image"])
        for _ in enumerate(data):
            pass
    assert "img should be NumPy image" in str(error_info.value)


def test_to_type_04():
    """
    Feature: ToType op
    Description: Test ToType op with missing output_type arg
    Expectation: Invalid input image type is detected.
    """

    logger.info("test_to_type_04")

    with pytest.raises(TypeError) as error_info:
        # Generate dataset
        data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
        transforms = [
            py_vision.Decode(),
            py_vision.ToTensor(),
            # Note: if output_type is not explicitly given
            py_vision.ToType()
        ]
        transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
        _ = data.map(operations=transform, input_columns=["image"])
    assert "missing" in str(error_info.value)


def test_to_type_05():
    """
    Feature: ToType op
    Description: Test ToType op with invalid output_type arg.
    Expectation: Invalid input is detected.
    """

    logger.info("test_to_type_05")

    with pytest.raises(RuntimeError) as error_info:
        # Generate dataset
        data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
        transforms = [
            py_vision.Decode(),
            py_vision.ToTensor(),
            # Note: if output_type is not valid
            py_vision.ToType('invalid')
        ]
        transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
        data = data.map(operations=transform, input_columns=["image"])
        for _ in enumerate(data):
            pass
    assert "data type" in str(error_info.value)


def test_to_type_invalid_arg():
    """
    Feature: ToType op
    Description: Test ToType op with invalid data_type arg.
    Expectation: Invalid input is detected.
    """

    with pytest.raises(TypeError) as error_info:
        # Generate dataset
        data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
        transforms = [
            py_vision.Decode(),
            py_vision.ToTensor(),
            # Note: if argument name is not correct
            py_vision.ToType(data_type="int32")
        ]
        transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
        _ = data.map(operations=transform, input_columns=["image"])
    assert "unexpected keyword argument" in str(error_info.value)


if __name__ == "__main__":
    test_to_type_op()
    test_to_type_01()
    test_to_type_02()
    test_to_type_03()
    test_to_type_04()
    test_to_type_05()
    test_to_type_invalid_arg()
