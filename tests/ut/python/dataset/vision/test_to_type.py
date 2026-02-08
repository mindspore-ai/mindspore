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
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger
from mindspore.dataset.core.datatypes import nptype_to_detype
from util import save_and_check_md5_pil

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
        vision.Decode(True),
        vision.ToTensor(),
        # Note: Convert the datatype from float32 to int16
        vision.ToType(np.int16)
    ]
    transform1 = mindspore.dataset.transforms.Compose(transforms1)
    data1 = data1.map(operations=transform1, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms2 = [
        vision.Decode(True),
        vision.ToTensor()
    ]
    transform2 = mindspore.dataset.transforms.Compose(transforms2)
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


def test_to_type_data_type():
    """
    Feature: ToType op
    Description: Test ToType op with mstype.dtype versus numpy.dtype data_type arg.
        Test ToType alias gives same results as TypeCast op.
    Expectation: Data results are correct and the same
    """

    # First dataset - Use default datatype float32
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms1 = [
        vision.Decode(True),
        vision.ToTensor()
    ]
    transform1 = mindspore.dataset.transforms.Compose(transforms1)
    data1 = data1.map(operations=transform1, input_columns=["image"])

    # Second dataset - Convert the datatype from float32 to nptype.int32
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms2 = [
        vision.Decode(True),
        vision.ToTensor(),
        # Note: Convert the datatype from float32 to np.int32.  Use explicit argument name.
        vision.ToType(data_type=np.int32)

    ]
    transform2 = mindspore.dataset.transforms.Compose(transforms2)
    data2 = data2.map(operations=transform2, input_columns=["image"])

    # Third dataset - Convert the datatype from float32 to mstype.int32
    data3 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms3 = [
        vision.Decode(True),
        vision.ToTensor(),
        # Note: Convert the datatype from float32 to mstype.int32.  Use explicit argument name.
        vision.ToType(data_type=mstype.int32)
    ]
    transform3 = mindspore.dataset.transforms.Compose(transforms3)
    data3 = data3.map(operations=transform3, input_columns=["image"])

    # Fourth dataset - Use TypeCast op. Convert the datatype from float32 to mstype.int32
    data4 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms4 = [
        vision.Decode(True),
        vision.ToTensor(),
        # Note: Convert the datatype from float32 to mstype.int32.  Use TypeCast op and argument name.
        mindspore.dataset.transforms.TypeCast(data_type=mstype.int32)
    ]
    transform4 = mindspore.dataset.transforms.Compose(transforms4)
    data4 = data4.map(operations=transform4, input_columns=["image"])

    for item1, item2, item3, item4 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                                          data2.create_dict_iterator(num_epochs=1, output_numpy=True),
                                          data3.create_dict_iterator(num_epochs=1, output_numpy=True),
                                          data4.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image1 = item1["image"]
        image2 = item2["image"]
        image3 = item3["image"]
        image4 = item4["image"]

        assert isinstance(image1, np.ndarray)
        assert isinstance(image2, np.ndarray)
        assert isinstance(image3, np.ndarray)
        assert isinstance(image4, np.ndarray)

        assert image1.dtype == np.float32
        assert image2.dtype == np.int32
        assert image3.dtype == np.int32
        assert image4.dtype == "int32"

        assert image1.shape == image2.shape
        assert image1.shape == image3.shape
        assert image1.shape == image4.shape


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
        vision.Decode(True),
        vision.ToTensor(),
        # Note: Convert the datatype from float32 to int32
        vision.ToType(np.int32)
    ]
    transform = mindspore.dataset.transforms.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])

    # Compare with expected md5 from images
    filename = "to_type_01_result.npz"
    save_and_check_md5_pil(data, filename, generate_golden=GENERATE_GOLDEN)


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
        vision.Decode(True),
        vision.ToTensor(),
        # Note: Convert to type int
        vision.ToType('int')
    ]
    transform = mindspore.dataset.transforms.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])

    # Compare with expected md5 from images
    filename = "to_type_02_result.npz"
    save_and_check_md5_pil(data, filename, generate_golden=GENERATE_GOLDEN)


def test_to_type_03():
    """
    Feature: ToType op
    Description: Test ToType op with PIL input to ToType op
    Expectation: Dataset pipeline runs successfully
    """

    logger.info("test_to_type_03")

    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        vision.Decode(True),
        # Note: No error PIL image input to ToType
        vision.ToType(np.int32)
    ]
    transform = mindspore.dataset.transforms.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])

    num_iter = 0
    for _ in enumerate(data):
        num_iter += 1
    assert num_iter == 3


def test_to_type_04():
    """
    Feature: ToType op
    Description: Test ToType op with missing data_type argument
    Expectation: Invalid input is detected.
    """
    logger.info("test_to_type_04")

    with pytest.raises(TypeError) as error_info:
        # Generate dataset
        data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
        transforms = [
            vision.Decode(True),
            vision.ToTensor(),
            # Note: if data_type is not explicitly given
            vision.ToType()
        ]
        transform = mindspore.dataset.transforms.Compose(transforms)
        _ = data.map(operations=transform, input_columns=["image"])
    assert "missing" in str(error_info.value)


def test_to_type_05():
    """
    Feature: ToType op
    Description: Test ToType op with invalid data_type arg.
    Expectation: Invalid input is detected.
    """
    logger.info("test_to_type_05")

    with pytest.raises(TypeError) as error_info:
        # Generate dataset
        data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
        transforms = [
            vision.Decode(True),
            vision.ToTensor(),
            # Note: if data_type is not valid
            vision.ToType('invalid')
        ]
        transform = mindspore.dataset.transforms.Compose(transforms)
        _ = data.map(operations=transform, input_columns=["image"])
    assert "Argument data_type with value invalid is not of type" in str(error_info.value)


def test_to_type_invalid_arg():
    """
    Feature: ToType op
    Description: Test ToType op with invalid output_type arg.
    Expectation: Invalid input is detected.
    """

    with pytest.raises(TypeError) as error_info:
        # Generate dataset
        data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
        transforms = [
            vision.Decode(True),
            vision.ToTensor(),
            # Note: if argument name is not correct
            vision.ToType(output_type="int32")
        ]
        transform = mindspore.dataset.transforms.Compose(transforms)
        _ = data.map(operations=transform, input_columns=["image"])
    assert "missing a required argument: 'data_type'" in str(error_info.value)


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


if __name__ == "__main__":
    test_to_type_op()
    test_to_type_data_type()
    test_to_type_01()
    test_to_type_02()
    test_to_type_03()
    test_to_type_04()
    test_to_type_05()
    test_to_type_invalid_arg()
    test_to_type_errors()
    test_np_to_de()
    test_to_type_operation_01()
    test_to_type_exception_01()
