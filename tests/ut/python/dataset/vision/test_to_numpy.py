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
Test ToNumpy op in Dataset
"""
import numpy as np
import os
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as vision
from util import config_get_set_seed

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
TEST_DATA_DATASET_FUNC ="../data/dataset/"
image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "jpg.jpg")


def test_to_numpy_op_1():
    """
    Feature: ToNumpy op
    Description: Test ToNumpy op converts to NumPy array and behaves like np.array
    Expectation: Data results are correct and the same
    """

    # First dataset with Decode(True) -> np.array
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms1 = [
        vision.Decode(True),
        # Note: Convert to NumPy array
        np.array
    ]
    data1 = data1.map(operations=transforms1, input_columns=["image"])

    # Second dataset with Decode(True) -> ToNumpy
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms2 = [
        vision.Decode(True),
        # Note: Convert to NumPy array
        vision.ToNumpy()
    ]
    data2 = data2.map(operations=transforms2, input_columns=["image"])

    for img1, img2 in zip(data1.create_tuple_iterator(num_epochs=1, output_numpy=True),
                          data2.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_equal(img1, img2)


def test_to_numpy_op_2():
    """
    Feature: ToNumpy op
    Description: Test ToNumpy op in data pipelines which are all equivalent
    Expectation: Data results are correct and the same
    """

    # First dataset with Decode(True) -> ToNumpy
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms1 = [
        vision.Decode(True),
        vision.ToNumpy()
    ]
    data1 = data1.map(operations=transforms1, input_columns=["image"])

    # Second dataset with Decode(True) -> ToNumpy -> ToPIL -> ToNumpy
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms2 = [
        vision.Decode(True),
        vision.ToNumpy(),
        vision.ToPIL(),
        vision.ToNumpy()
    ]
    data2 = data2.map(operations=transforms2, input_columns=["image"])

    # Third dataset - without ToNumpy
    data3 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms3 = [
        vision.Decode(True)
    ]
    data3 = data3.map(operations=transforms3, input_columns=["image"])

    for img1, img2, img3 in zip(data1.create_tuple_iterator(num_epochs=1, output_numpy=True),
                                data2.create_tuple_iterator(num_epochs=1, output_numpy=True),
                                data3.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_equal(img1, img2)
        np.testing.assert_equal(img1, img3)


def test_to_numpy_op_3():
    """
    Feature: ToNumpy op
    Description: Test ToNumpy op in data pipeline to select C++ implementation of subsequent op
    Expectation: Data results are correct and the same
    """
    original_seed = config_get_set_seed(10)

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms1 = [
        vision.Decode(True),
        vision.RandomHorizontalFlip(1.0),  # Python implementation selected
        vision.ToNumpy()
    ]
    data1 = data1.map(operations=transforms1, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms2 = [
        vision.Decode(True),
        vision.ToNumpy(),
        vision.HorizontalFlip()  # Only C++ implementation available
    ]
    data2 = data2.map(operations=transforms2, input_columns=["image"])

    # Third dataset
    data3 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms3 = [
        vision.Decode(True),
        vision.ToNumpy(),
        vision.RandomHorizontalFlip(1.0)  # C++ implementation selected
    ]
    data3 = data3.map(operations=transforms3, input_columns=["image"])

    for img1, img2, img3 in zip(data1.create_tuple_iterator(num_epochs=1, output_numpy=True),
                                data2.create_tuple_iterator(num_epochs=1, output_numpy=True),
                                data3.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_equal(img1, img2)
        np.testing.assert_equal(img1, img3)

    # Restore configuration
    ds.config.set_seed(original_seed)


def test_tonumpy_operation_01():
    """
    Feature: ToNumpy operation
    Description: Testing the normal functionality of the ToNumpy operator
    Expectation: The Output is equal to the expected output
    """
    # ToNumpy input Pillow, eager mode
    image = Image.open(image_jpg)
    totensor_op = vision.ToNumpy()
    out = totensor_op(image)
    assert isinstance(out, np.ndarray)
    assert (out == np.array(image)).all()

    # To convert Pillow to NumPy in eager mode, first convert Pillow to NumPy, then convert NumPy back to Pillow.
    image = np.random.randint(0, 255, (32, 32, 3)).astype(np.uint8)
    topil_op = vision.ToPIL()
    pil = topil_op(image)
    np_ops = vision.ToNumpy()
    out = np_ops(pil)
    assert isinstance(out, np.ndarray)
    assert (out == image).all()

    # ToNumpy input numpy, eager mode, 4-dimensional
    image = np.random.randn(32, 32, 3, 3)
    np_ops = vision.ToNumpy()
    out = np_ops(image)
    assert isinstance(out, np.ndarray)
    assert (out == image).all()

    # ToNumpy input numpy, eager mode, 2-dimensional
    image = list(np.random.randint(0, 255, (32, 32, 3, 3)).astype(np.int32))
    np_ops = vision.ToNumpy()
    out = np_ops(image)
    assert isinstance(out, np.ndarray)
    assert (out == image).all()

    # ToNumpy input numpy, eager mode, 1-dimensional
    image = (1.5, 2.5, 3.0, 4)
    np_ops = vision.ToNumpy()
    out = np_ops(image)
    assert isinstance(out, np.ndarray)
    assert (out == image).all()

    # ToNumpy input number
    image = 10
    np_ops = vision.ToNumpy()
    out = np_ops(image)
    assert isinstance(out, np.ndarray)
    assert (out == image).all()

    # Converting a list of string type to Numpy
    image = ["a", "b", "c"]
    np_ops = vision.ToNumpy()
    out = np_ops(image)
    assert isinstance(out, np.ndarray)
    assert (out == image).all()


def test_tonumpy_exception_01():
    """
    Feature: ToNumpy operation
    Description: Testing the ToNumpy Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Passing a non-existent parameter to NumPy
    with pytest.raises(TypeError, match="__init__\\(\\) takes 1 positional argument but 2 were given"):
        vision.ToNumpy(10)


if __name__ == "__main__":
    test_to_numpy_op_1()
    test_to_numpy_op_2()
    test_to_numpy_op_3()
    test_tonumpy_operation_01()
    test_tonumpy_exception_01()
