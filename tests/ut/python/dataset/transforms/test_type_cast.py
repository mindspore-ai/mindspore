# Copyright 2019 Huawei Technologies Co., Ltd
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
Testing TypeCast op in DE
"""
import numpy as np
import pytest

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms as data_trans
from mindspore.dataset import vision
from mindspore import log as logger

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_type_cast():
    """
    Feature: TypeCast op
    Description: Test TypeCast op with mstype.float32 data_type arg.
    Expectation: Data results are correct
    """

    logger.info("test_type_cast")

    # First dataset - Use TypeCast with mindspore datatype
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    type_cast_op = data_trans.TypeCast(data_type=mstype.float32)
    ctrans = [decode_op,
              type_cast_op,
              ]
    data1 = data1.map(operations=ctrans, input_columns=["image"])

    # Second dataset
    transforms = [vision.Decode(True),
                  vision.ToTensor()
                  ]
    transform = data_trans.Compose(transforms)
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=transform, input_columns=["image"])

    # Third dataset - Use TypeCast with Numpy datatype
    data3 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    trans3 = [vision.Decode(),
              data_trans.TypeCast(data_type=np.float32)]
    data3 = data3.map(operations=trans3, input_columns=["image"])

    num_iter = 0
    for item1, item2, item3 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                                   data2.create_dict_iterator(num_epochs=1, output_numpy=True),
                                   data3.create_dict_iterator(num_epochs=1, output_numpy=True)):
        num_iter += 1
        c_image = item1["image"]
        image = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image3 = item3["image"]

        assert c_image.dtype == "float32"
        assert image3.dtype == "float32"

        assert isinstance(c_image, np.ndarray)
        assert isinstance(image, np.ndarray)
        assert isinstance(image3, np.ndarray)

        assert c_image.shape == image.shape
        assert image3.shape == image.shape


def test_type_cast_string():
    """
    Feature: TypeCast op
    Description: Test TypeCast op with string "float16" data_type arg.
    Expectation: Data results are correct
    """

    logger.info("test_type_cast_string")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    type_cast_op = data_trans.TypeCast(data_type="float16")
    ctrans = [decode_op,
              type_cast_op
              ]
    data1 = data1.map(operations=ctrans, input_columns=["image"])

    # Second dataset
    transforms = [vision.Decode(True),
                  vision.ToTensor()
                  ]
    transform = data_trans.Compose(transforms)
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=transform, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        num_iter += 1
        c_image = item1["image"]
        image = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)

        logger.info("shape of c_image: {}".format(c_image.shape))
        logger.info("shape of image: {}".format(image.shape))

        logger.info("dtype of c_image: {}".format(c_image.dtype))
        logger.info("dtype of image: {}".format(image.dtype))
        assert c_image.dtype == "float16"

        assert isinstance(c_image, np.ndarray)
        assert isinstance(image, np.ndarray)

        assert c_image.shape == image.shape


def test_type_cast_eager():
    """
    Feature: Test eager of TypeCast
    Description: Cast from string to string / from int to bool / from float to int
    Expectation: Cast successfully
    """
    type_cast_op1 = data_trans.TypeCast(mstype.string)
    result1 = type_cast_op1("test_strings")
    assert result1 == "test_strings"

    type_cast_op2 = data_trans.TypeCast(mstype.bool_)
    result2 = type_cast_op2(0)
    assert result2.tolist() is False

    type_cast_op3 = data_trans.TypeCast(mstype.int32)
    result3 = type_cast_op3([1.131613, 0.12415, 68.88])
    assert result3.tolist() == [1, 0, 68]


def test_type_cast_exception():
    """
    Feature: Test exception TypeCast
    Description: Test exception TypeCast
    Expectation: Fail as expectation
    """

    def gen():
        for _ in range(1):
            yield np.array(["aaaa", "bbbb", "cccc"])

    with pytest.raises(RuntimeError):
        dataset = ds.GeneratorDataset(gen, ["data1"])
        type_cast_op1 = data_trans.TypeCast(mstype.bool_)
        dataset = dataset.map(type_cast_op1, input_columns=["data1"])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    with pytest.raises(RuntimeError):
        type_cast_op2 = data_trans.TypeCast(mstype.string)
        _ = type_cast_op2([1, 2, 3, 4])


def test_type_cast_invalid_arg():
    """
    Feature: TypeCast op
    Description: Test TypeCast op with invalid output_type arg.
    Expectation: Invalid input is detected.
    """

    with pytest.raises(TypeError) as error_info:
        # Generate dataset
        data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
        transforms = [
            vision.Decode(True),
            vision.ToTensor(),
            # Note: if argument name is not correct
            data_trans.TypeCast(output_type="float32")
        ]
        transform = data_trans.Compose(transforms)
        _ = data.map(operations=transform, input_columns=["image"])
    assert "missing a required argument: 'data_type'" in str(error_info.value)


def test_type_cast_err_missing_arg():
    """
    Feature: TypeCast op
    Description: Test TypeCast op with missing data_type argument
    Expectation: Invalid input is detected.
    """
    logger.info("test_type_cast_err_missing_arg")

    with pytest.raises(TypeError) as error_info:
        # Generate dataset
        data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
        transforms = [
            vision.Decode(True),
            vision.ToTensor(),
            # Note: if data_type is not explicitly given
            data_trans.TypeCast()
        ]
        transform = data_trans.Compose(transforms)
        _ = data.map(operations=transform, input_columns=["image"])
    assert "missing a required argument: 'data_type'" in str(error_info.value)


def test_type_cast_err_invalid_arg():
    """
    Feature: TypeCast op
    Description: Test TypeCast op with invalid data_type arg.
    Expectation: Invalid input is detected.
    """
    logger.info("test_type_cast_err_invalid_arg")

    with pytest.raises(TypeError) as error_info:
        # Generate dataset
        data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
        transforms = [
            vision.Decode(True),
            vision.ToTensor(),
            # Note: if data_type is not valid
            data_trans.TypeCast("junk")
        ]
        transform = data_trans.Compose(transforms)
        _ = data.map(operations=transform, input_columns=["image"])
    assert "Argument data_type with value junk is not of type" in str(error_info.value)


def test_type_cast_errors():
    """
    Feature: TypeCast op
    Description: Test TypeCast with invalid input
    Expectation: Correct error is thrown as expected
    """
    with pytest.raises(TypeError) as error_info:
        data_trans.TypeCast("JUNK")
    assert "Argument data_type with value JUNK is not of type" in str(error_info.value)

    with pytest.raises(TypeError) as error_info:
        data_trans.TypeCast([np.int32])
    assert "Argument data_type with value [<class 'numpy.int32'>] is not of type" in str(error_info.value)

    with pytest.raises(TypeError) as error_info:
        data_trans.TypeCast((np.int32,))
    assert "Argument data_type with value (<class 'numpy.int32'>,) is not of type" in str(error_info.value)

    with pytest.raises(TypeError) as error_info:
        data_trans.TypeCast((np.float16, np.int8))
    assert "Argument data_type with value (<class 'numpy.float16'>, <class 'numpy.int8'>) is not of type" \
           in str(error_info.value)

    with pytest.raises(TypeError) as error_info:
        data_trans.TypeCast(None)
    assert "Argument data_type with value None is not of type" in str(error_info.value)


if __name__ == "__main__":
    test_type_cast()
    test_type_cast_string()
    test_type_cast_eager()
    test_type_cast_exception()
    test_type_cast_invalid_arg()
    test_type_cast_err_missing_arg()
    test_type_cast_err_invalid_arg()
    test_type_cast_errors()
