# Copyright 2020 Huawei Technologies Co., Ltd
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
import mindspore._c_dataengine as cde
import mindspore.dataset as ds
import mindspore.dataset.transforms.transforms
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger
from mindspore.dataset.core.datatypes import nptype_to_detype
from util import save_and_check_md5

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_to_type_op():
    """
    Test ToType Op
    """
    logger.info("test_to_type_op")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms1 = [
        vision.Decode(True),
        vision.ToTensor(),
        # Note: Convert the datatype from float32 to int16
        vision.ToType(np.int16)
    ]
    transform1 = mindspore.dataset.transforms.transforms.Compose(transforms1)
    data1 = data1.map(operations=transform1, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms2 = [
        vision.Decode(True),
        vision.ToTensor()
    ]
    transform2 = mindspore.dataset.transforms.transforms.Compose(transforms2)
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
    Test ToType Op with md5 comparison: valid input (Numpy dtype)
    Expect to pass
    """
    logger.info("test_to_type_01")

    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        vision.Decode(True),
        vision.ToTensor(),
        # Note: Convert the datatype from float32 to int32
        vision.ToType(np.int32)
    ]
    transform = mindspore.dataset.transforms.transforms.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])

    # Compare with expected md5 from images
    filename = "to_type_01_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)


def test_to_type_02():
    """
    Test ToType Op with md5 comparison: valid input (str)
    Expect to pass
    """
    logger.info("test_to_type_02")
    # Generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        vision.Decode(True),
        vision.ToTensor(),
        # Note: Convert to type int
        vision.ToType('int')
    ]
    transform = mindspore.dataset.transforms.transforms.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])

    # Compare with expected md5 from images
    filename = "to_type_02_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)


def test_to_type_03():
    """
    Test ToType Op: invalid input image type
    Expect to raise error
    """
    logger.info("test_to_type_03")

    try:
        # Generate dataset
        data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
        transforms = [
            vision.Decode(True),
            # Note: If the object is not numpy, e.g. PIL image, TypeError will raise
            vision.ToType(np.int32)
        ]
        transform = mindspore.dataset.transforms.transforms.Compose(transforms)
        data = data.map(operations=transform, input_columns=["image"])
    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Numpy" in str(e)


def test_to_type_04():
    """
    Test ToType Op: no output_type given
    Expect to raise error
    """
    logger.info("test_to_type_04")

    try:
        # Generate dataset
        data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
        transforms = [
            vision.Decode(True),
            vision.ToTensor(),
            # Note: if output_type is not explicitly given
            vision.ToType()
        ]
        transform = mindspore.dataset.transforms.transforms.Compose(transforms)
        data = data.map(operations=transform, input_columns=["image"])
    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "missing" in str(e)


def test_to_type_05():
    """
    Test ToType Op: invalid output_type
    Expect to raise error
    """
    logger.info("test_to_type_05")

    try:
        # Generate dataset
        data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
        transforms = [
            vision.Decode(True),
            vision.ToTensor(),
            # Note: if output_type is not explicitly given
            vision.ToType('invalid')
        ]
        transform = mindspore.dataset.transforms.transforms.Compose(transforms)
        data = data.map(operations=transform, input_columns=["image"])
    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "data type" in str(e)


def test_np_to_de():
    """
    Feature: NumPy Type to DE Type conversion
    Description: Test NumPy Type to DE Type conversion for all valid types
    Expectation: Data results are correct
    """

    assert nptype_to_detype(np.dtype("bool")) == cde.DataType("bool")

    assert nptype_to_detype(np.dtype("int8")) == cde.DataType("int8")
    assert nptype_to_detype(np.dtype("int16")) == cde.DataType("int16")
    assert nptype_to_detype(np.dtype("int32")) == cde.DataType("int32")
    assert nptype_to_detype(np.dtype("int64")) == cde.DataType("int64")
    assert nptype_to_detype(np.dtype("int")) == cde.DataType("int64")

    assert nptype_to_detype(np.dtype("uint8")) == cde.DataType("uint8")
    assert nptype_to_detype(np.dtype("uint16")) == cde.DataType("uint16")
    assert nptype_to_detype(np.dtype("uint32")) == cde.DataType("uint32")
    assert nptype_to_detype(np.dtype("uint64")) == cde.DataType("uint64")

    assert nptype_to_detype(np.dtype("float16")) == cde.DataType("float16")
    assert nptype_to_detype(np.dtype("float32")) == cde.DataType("float32")
    assert nptype_to_detype(np.dtype("float64")) == cde.DataType("float64")

    assert nptype_to_detype(np.dtype("str")) == cde.DataType("string")

    assert nptype_to_detype(bool) == cde.DataType("bool")

    assert nptype_to_detype(np.int8) == cde.DataType("int8")
    assert nptype_to_detype(np.int16) == cde.DataType("int16")
    assert nptype_to_detype(np.int32) == cde.DataType("int32")
    assert nptype_to_detype(np.int64) == cde.DataType("int64")
    assert nptype_to_detype(int) == cde.DataType("int64")

    assert nptype_to_detype(np.uint8) == cde.DataType("uint8")
    assert nptype_to_detype(np.uint16) == cde.DataType("uint16")
    assert nptype_to_detype(np.uint32) == cde.DataType("uint32")
    assert nptype_to_detype(np.uint64) == cde.DataType("uint64")

    assert nptype_to_detype(np.float16) == cde.DataType("float16")
    assert nptype_to_detype(np.float32) == cde.DataType("float32")
    assert nptype_to_detype(np.float64) == cde.DataType("float64")

    assert nptype_to_detype(str) == cde.DataType("string")


if __name__ == "__main__":
    test_to_type_op()
    test_to_type_01()
    test_to_type_02()
    test_to_type_03()
    test_to_type_04()
    test_to_type_05()
    test_np_to_de()
