# Copyright 2022 Huawei Technologies Co., Ltd
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
Testing ToTensor op in DE
"""
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.vision.py_transforms as py_vision

DATA_DIR = "../data/dataset/testMnistData"

DATA_DIR_TF = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR_TF = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_to_tensor_float32():
    """
    Feature: ToTensor Op
    Description: Test with default output_type
    Expectation: Dataset pipeline runs successfully and results are verified
    """
    data1 = ds.MnistDataset(DATA_DIR, num_samples=10, shuffle=False)

    data2 = ds.MnistDataset(DATA_DIR, num_samples=10, shuffle=False)

    data2 = data2.map(operations=[py_vision.ToTensor()], num_parallel_workers=1)

    for d1, d2 in zip(data1.create_tuple_iterator(num_epochs=1, output_numpy=True),
                      data2.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        img1, img2 = d1[0], d2[0]
        img1 = img1 / 255
        img1 = np.transpose(img1, (2, 0, 1))

        np.testing.assert_almost_equal(img2, img1, 5)


def test_to_tensor_float64():
    """
    Feature: ToTensor Op
    Description: Test with np.float64 output_type
    Expectation: Dataset pipeline runs successfully and results are verified
    """
    data1 = ds.MnistDataset(DATA_DIR, num_samples=10, shuffle=False)

    data2 = ds.MnistDataset(DATA_DIR, num_samples=10, shuffle=False)

    data2 = data2.map(operations=[py_vision.ToTensor(np.float64)], num_parallel_workers=1)

    for d1, d2 in zip(data1.create_tuple_iterator(num_epochs=1, output_numpy=True),
                      data2.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        img1, img2 = d1[0], d2[0]
        img1 = img1 / 255
        img1 = np.transpose(img1, (2, 0, 1))

        np.testing.assert_almost_equal(img2, img1, 5)


def test_to_tensor_int32():
    """
    Feature: ToTensor Op
    Description: Test with np.int32 output_type
    Expectation: Dataset pipeline runs successfully and results are verified
    """
    data1 = ds.MnistDataset(DATA_DIR, num_samples=10, shuffle=False)

    data2 = ds.MnistDataset(DATA_DIR, num_samples=10, shuffle=False)

    data2 = data2.map(operations=[py_vision.ToTensor(np.int32)], num_parallel_workers=1)

    for d1, d2 in zip(data1.create_tuple_iterator(num_epochs=1, output_numpy=True),
                      data2.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        img1, img2 = d1[0], d2[0]
        img1 = img1 / 255
        img1 = img1.astype('int')
        img1 = np.transpose(img1, (2, 0, 1))

        np.testing.assert_almost_equal(img2, img1, 5)


def test_to_tensor_eager():
    """
    Feature: ToTensor Op
    Description: Test with various output_type in eager scenario with float16 image
    Expectation: Test runs successfully and results are verified
    """

    def test_config(my_np_type):
        image = np.random.randn(128, 128, 3).astype(np.float16)
        op = py_vision.ToTensor(output_type=my_np_type)
        out = op(image)

        image = image / 255
        image = image.astype(my_np_type)
        image = np.transpose(image, (2, 0, 1))

        np.testing.assert_almost_equal(out, image, 5)

    test_config(np.float16)
    test_config(np.float32)
    test_config(np.float64)


def test_to_tensor_float16():
    """
    Feature: ToTensor Op
    Description: Test with float16 output_type
    Expectation: Dataset pipeline runs successfully and results are verified
    """
    data1 = ds.MnistDataset(DATA_DIR, num_samples=10, shuffle=False)

    data2 = ds.MnistDataset(DATA_DIR, num_samples=10, shuffle=False)
    # For ToTensor, use ms_type float16 output_type
    data2 = data2.map(operations=[py_vision.ToTensor("float16")], num_parallel_workers=1)

    for d1, d2 in zip(data1.create_tuple_iterator(num_epochs=1, output_numpy=True),
                      data2.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        img1, img2 = d1[0], d2[0]
        img1 = img1 / 255
        img1 = np.transpose(img1, (2, 0, 1))

        np.testing.assert_almost_equal(img2, img1, 3)


def test_to_tensor_eager_float16():
    """
    Feature: ToTensor Op
    Description: Test ToTensor with various supported output_type values in eager scenario with float16 image
    Expectation: Test runs successfully and results are verified
    """

    def test_config(my_output_type, output_dtype):
        image = np.random.randn(128, 128, 3).astype(np.float16)
        op = py_vision.ToTensor(output_type=my_output_type)
        out = op(image)

        assert out.dtype == output_dtype

        image = image / 255
        image = image.astype(my_output_type)
        image = np.transpose(image, (2, 0, 1))

        np.testing.assert_almost_equal(out, image, 5)

    test_config(np.float16, "float16")
    test_config(np.float32, "float32")
    test_config(np.float64, "float64")
    test_config(np.int8, "int8")
    test_config(np.int16, "int16")
    test_config(np.int32, "int32")
    test_config(np.int64, "int64")
    test_config(np.uint8, "uint8")
    test_config(np.uint16, "uint16")
    test_config(np.uint32, "uint32")
    test_config(np.uint64, "uint64")
    test_config(np.bool, "bool")


def test_to_tensor_eager_float64():
    """
    Feature: ToTensor Op
    Description: Test ToTensor with various supported output_type values in eager scenario with float64 image
    Expectation: Test runs successfully and results are verified
    """

    def test_config(my_output_type, output_dtype):
        image = np.random.randn(128, 128, 3).astype(np.float64)
        op = py_vision.ToTensor(output_type=my_output_type)
        out = op(image)

        assert out.dtype == output_dtype

        image = image / 255
        image = image.astype(my_output_type)
        image = np.transpose(image, (2, 0, 1))

        np.testing.assert_almost_equal(out, image, 5)

    test_config(np.float16, "float16")
    test_config(np.float32, "float32")
    test_config(np.float64, "float64")
    test_config(np.int8, "int8")
    test_config(np.int16, "int16")
    test_config(np.int32, "int32")
    test_config(np.int64, "int64")
    test_config(np.uint8, "uint8")
    test_config(np.uint16, "uint16")
    test_config(np.uint32, "uint32")
    test_config(np.uint64, "uint64")
    test_config(np.bool, "bool")


def test_to_tensor_eager_int32():
    """
    Feature: ToTensor Op
    Description: Test ToTensor with various supported output_type values in eager scenario with int32 image
    Expectation: Test runs successfully and results are verified
    """

    def test_config(my_output_type, output_dtype):
        image = np.random.randn(128, 128, 3).astype(np.int32)
        op = py_vision.ToTensor(output_type=my_output_type)
        out = op(image)

        assert out.dtype == output_dtype

        image = image / 255
        image = image.astype(my_output_type)
        image = np.transpose(image, (2, 0, 1))

        np.testing.assert_almost_equal(out, image, 5)

    test_config(np.float16, "float16")
    test_config(np.float32, "float32")
    test_config(np.float64, "float64")
    test_config(np.int8, "int8")
    test_config(np.int16, "int16")
    test_config(np.int32, "int32")
    test_config(np.int64, "int64")
    test_config(np.uint8, "uint8")
    test_config(np.uint16, "uint16")
    test_config(np.uint32, "uint32")
    test_config(np.uint64, "uint64")
    test_config(np.bool, "bool")


def test_to_tensor_eager_int64():
    """
    Feature: ToTensor Op
    Description: Test ToTensor with various supported output_type values in eager scenario with int64 image
    Expectation: Test runs successfully and results are verified
    """

    def test_config(my_output_type, output_dtype):
        image = np.random.randn(128, 128, 3).astype(np.int64)
        op = py_vision.ToTensor(output_type=my_output_type)
        out = op(image)

        assert out.dtype == output_dtype

        image = image / 255
        image = image.astype(my_output_type)
        image = np.transpose(image, (2, 0, 1))

        np.testing.assert_almost_equal(out, image, 5)

    test_config(np.float16, "float16")
    test_config(np.float32, "float32")
    test_config(np.float64, "float64")
    test_config(np.int8, "int8")
    test_config(np.int16, "int16")
    test_config(np.int32, "int32")
    test_config(np.int64, "int64")
    test_config(np.uint8, "uint8")
    test_config(np.uint16, "uint16")
    test_config(np.uint32, "uint32")
    test_config(np.uint64, "uint64")
    test_config(np.bool, "bool")


def test_to_tensor_eager_uint32():
    """
    Feature: ToTensor Op
    Description: Test ToTensor with various supported output_type values in eager scenario with uint32 image
    Expectation: Test runs successfully and results are verified
    """

    def test_config(my_output_type, output_dtype):
        image = np.random.randn(128, 128, 3).astype(np.uint32)
        op = py_vision.ToTensor(output_type=my_output_type)
        out = op(image)

        assert out.dtype == output_dtype

        image = image / 255
        image = image.astype(my_output_type)
        image = np.transpose(image, (2, 0, 1))

        np.testing.assert_almost_equal(out, image, 5)

    test_config(np.float16, "float16")
    test_config(np.float32, "float32")
    test_config(np.float64, "float64")
    test_config(np.int8, "int8")
    test_config(np.int16, "int16")
    test_config(np.int32, "int32")
    test_config(np.int64, "int64")
    test_config(np.uint8, "uint8")
    test_config(np.uint16, "uint16")
    test_config(np.uint32, "uint32")
    test_config(np.uint64, "uint64")
    test_config(np.bool, "bool")


def test_to_tensor_eager_uint64():
    """
    Feature: ToTensor Op
    Description: Test ToTensor with various supported output_type values in eager scenario with uint64 image
    Expectation: Test runs successfully and results are verified
    """

    def test_config(my_output_type, output_dtype):
        image = np.random.randn(128, 128, 3).astype(np.uint64)
        op = py_vision.ToTensor(output_type=my_output_type)
        out = op(image)

        assert out.dtype == output_dtype

        image = image / 255
        image = image.astype(my_output_type)
        image = np.transpose(image, (2, 0, 1))

        np.testing.assert_almost_equal(out, image, 5)

    test_config(np.float16, "float16")
    test_config(np.float32, "float32")
    test_config(np.float64, "float64")
    test_config(np.int8, "int8")
    test_config(np.int16, "int16")
    test_config(np.int32, "int32")
    test_config(np.int64, "int64")
    test_config(np.uint8, "uint8")
    test_config(np.uint16, "uint16")
    test_config(np.uint32, "uint32")
    test_config(np.uint64, "uint64")
    test_config(np.bool, "bool")


def test_to_tensor_eager_bool():
    """
    Feature: ToTensor Op
    Description: Test ToTensor in eager scenario with bool image
    Expectation: Test runs successfully and results are verified
    """

    image = np.random.randint(0, 255, (128, 128, 3)).astype(np.bool)
    my_np_type = np.uint8
    op = py_vision.ToTensor(output_type=my_np_type)
    out = op(image)

    assert out.dtype == "uint8"


def test_to_tensor_errors():
    """
    Feature: ToTensor op
    Description: Test ToTensor with invalid input
    Expectation: Correct error is thrown as expected
    """

    # Test wrong parameter name
    with pytest.raises(TypeError) as error_info:
        py_vision.ToTensor(data_type=np.int16)
    assert "got an unexpected keyword argument 'data_type'" in str(error_info.value)


def skip_test_to_tensor_errors2():
    """
    Feature: ToTensor op
    Description: Test ToTensor with invalid input
    Expectation: Correct error is thrown as expected
    """
    # Note:All these scenarios do not raise a TypeError due to no validator for py_vision.ToTensor
    with pytest.raises(TypeError) as error_info:
        py_vision.ToTensor("JUNK")
    assert "Argument output_type with value JUNK is not of type" in str(error_info.value)

    with pytest.raises(TypeError) as error_info:
        py_vision.ToTensor([np.float64])
    assert "Argument output_type with value [<class 'numpy.float64'>] is not of type" in str(error_info.value)

    with pytest.raises(TypeError) as error_info:
        py_vision.ToTensor((np.float64,))
    assert "Argument output_type with value (<class 'numpy.float64'>,) is not of type" in str(error_info.value)

    with pytest.raises(TypeError) as error_info:
        py_vision.ToTensor((np.float16, np.int8))
    assert "Argument output_type with value (<class 'numpy.float16'>, <class 'numpy.int8'>) is not of type" \
           in str(error_info.value)

    with pytest.raises(TypeError) as error_info:
        py_vision.ToTensor(None)
    assert "Argument output_type with value None is not of type" in str(error_info.value)


def test_to_tensor_eager_error_string():
    """
    Feature: ToTensor op
    Description: Test ToTensor in eager scenario with string image
    Expectation: Correct error is thrown as expected
    """
    image = np.random.randint(0, 255, (128, 128, 3)).astype(np.str)
    my_np_type = np.uint8
    with pytest.raises(TypeError) as error_info:
        op = py_vision.ToTensor(output_type=my_np_type)
        _ = op(image)
    assert "ufunc 'true_divide' not supported for the input types" in str(error_info.value)


if __name__ == "__main__":
    test_to_tensor_float32()
    test_to_tensor_float64()
    test_to_tensor_int32()
    test_to_tensor_eager()
    test_to_tensor_float16()
    test_to_tensor_eager_float16()
    test_to_tensor_eager_float64()
    test_to_tensor_eager_int32()
    test_to_tensor_eager_int64()
    test_to_tensor_eager_uint32()
    test_to_tensor_eager_uint64()
    test_to_tensor_eager_int32()
    test_to_tensor_eager_bool()
    test_to_tensor_errors()
    skip_test_to_tensor_errors2()
    test_to_tensor_eager_error_string()
