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
Testing ToType op in DE
"""
import numpy as np
import pytest
import mindspore as ms
import mindspore._c_dataengine as cde
import mindspore.dataset.vision.transforms as vision
from mindspore.dataset.core.datatypes import nptype_to_detype

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


















def test_to_type_errors():
    """
    Feature: ToType op
    Description: Test ToType with invalid input
    Expectation: Correct error is thrown as expected
    """
    with pytest.raises(TypeError) as error_info:
        vision.ToType("JUNK")
    assert "Argument data_type with value JUNK is not of type" in str(error_info.value)

    with pytest.raises(TypeError) as error_info:
        vision.ToType([np.float32])
    assert "Argument data_type with value [<class 'numpy.float32'>] is not of type" in str(error_info.value)

    with pytest.raises(TypeError) as error_info:
        vision.ToType((np.float32,))
    assert "Argument data_type with value (<class 'numpy.float32'>,) is not of type" in str(error_info.value)

    with pytest.raises(TypeError) as error_info:
        vision.ToType((np.int16, np.int8))
    assert "Argument data_type with value (<class 'numpy.int16'>, <class 'numpy.int8'>) is not of type" \
           in str(error_info.value)

    with pytest.raises(TypeError) as error_info:
        vision.ToType(None)
    assert "Argument data_type with value None is not of type" in str(error_info.value)


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


def test_to_type_operation_01():
    """
    Feature: ToType operation
    Description: Testing the normal functionality of the ToType operator
    Expectation: The Output is equal to the expected output
    """
    # ToType operation: np.float16, eager mode
    image = np.random.randint(0, 255, (20, 30, 8)).astype(np.uint8)
    totype_op = vision.ToType(np.float16)
    out = totype_op(image)
    assert out.dtype == "float16"

    # ToType operation: np.float64 to ms.int16
    image = np.random.randn(20, 30).astype(np.float64)
    totype_op = vision.ToType(ms.int16)
    out = totype_op(image)
    assert out.dtype == "int16"

    # ToType operation: int to bool
    image = np.random.randint(-1000, 1000, (10, 6, 8, 5)).astype(np.int32)
    totype_op = vision.ToType(np.bool_)
    out = totype_op(image)
    assert out.dtype == "bool"

    # ToType operation: np.float64 to uint8
    image = np.random.randn(50, )
    totype_op = vision.ToType(ms.uint8)
    out = totype_op(image)
    assert out.dtype == "uint8"

    # ToType operation: np.int64 to ms.int64
    image = tuple(np.random.randn(5, 4, 8, 12, 3).astype(np.int64))
    totype_op = vision.ToType(ms.int64)
    out = totype_op(image)
    assert out.dtype == "int64"

    # ToType operation: input equals list
    image = list(np.random.randn(100, 200).astype(np.int8))
    totype_op = vision.ToType(int)
    out = totype_op(image)
    assert out.dtype in ["int64", "int32"]

    # ToType operation: str t0 str
    image = np.array(["1", "2", "3"])
    totype_op = vision.ToType(ms.string)
    totype_op(image)

    # ToType operation: input equals 10.0
    image = 10.0
    totype_op = vision.ToType(np.int32)
    out = totype_op(image)
    assert out == 10


def test_to_type_exception_01():
    """
    Feature: ToType operation
    Description: Testing the ToType Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # ToType operation: float to int
    image = np.random.randn(100, 200)
    image = vision.ToPIL()(image)
    totype_op = vision.ToType(np.int32)
    totype_op(image)

    # ToType operation: Missing parameters
    with pytest.raises(TypeError, match="missing a required argument: 'data_type'"):
        vision.ToType()

    # ToType operation: The parameter is a list
    image = np.random.randn(100, 200).astype(np.float32)
    with pytest.raises(TypeError, match="but got <class 'list'>."):
        totype_op = vision.ToType([np.int32])
        totype_op(image)

    # ToType operation: The parameter is a int
    with pytest.raises(TypeError, match="but got <class 'int'>."):
        vision.ToType(1)

    # ToType operation: The parameter is a str
    with pytest.raises(TypeError, match="but got <class 'str'>."):
        vision.ToType("number")
