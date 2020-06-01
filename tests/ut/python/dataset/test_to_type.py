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
import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.py_transforms as py_vision
from mindspore import log as logger
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
        py_vision.Decode(),
        py_vision.ToTensor(),
        # Note: Convert the datatype from float32 to int16
        py_vision.ToType(np.int16)
    ]
    transform1 = py_vision.ComposeOp(transforms1)
    data1 = data1.map(input_columns=["image"], operations=transform1())

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms2 = [
        py_vision.Decode(),
        py_vision.ToTensor()
    ]
    transform2 = py_vision.ComposeOp(transforms2)
    data2 = data2.map(input_columns=["image"], operations=transform2())

    for item1, item2 in zip(data1.create_dict_iterator(), data2.create_dict_iterator()):
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
        py_vision.Decode(),
        py_vision.ToTensor(),
        # Note: Convert the datatype from float32 to int32
        py_vision.ToType(np.int32)
    ]
    transform = py_vision.ComposeOp(transforms)
    data = data.map(input_columns=["image"], operations=transform())

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
        py_vision.Decode(),
        py_vision.ToTensor(),
        # Note: Convert to type int
        py_vision.ToType('int')
    ]
    transform = py_vision.ComposeOp(transforms)
    data = data.map(input_columns=["image"], operations=transform())

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
            py_vision.Decode(),
            # Note: If the object is not numpy, e.g. PIL image, TypeError will raise
            py_vision.ToType(np.int32)
        ]
        transform = py_vision.ComposeOp(transforms)
        data = data.map(input_columns=["image"], operations=transform())
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
            py_vision.Decode(),
            py_vision.ToTensor(),
            # Note: if output_type is not explicitly given
            py_vision.ToType()
        ]
        transform = py_vision.ComposeOp(transforms)
        data = data.map(input_columns=["image"], operations=transform())
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
            py_vision.Decode(),
            py_vision.ToTensor(),
            # Note: if output_type is not explicitly given
            py_vision.ToType('invalid')
        ]
        transform = py_vision.ComposeOp(transforms)
        data = data.map(input_columns=["image"], operations=transform())
    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "data type" in str(e)

if __name__ == "__main__":
    test_to_type_op()
    test_to_type_01()
    test_to_type_02()
    test_to_type_03()
    test_to_type_04()
    test_to_type_05()
