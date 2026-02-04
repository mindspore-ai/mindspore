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
Testing LinearTransformation op in DE
"""
import numpy as np
import os
import pytest
from PIL import Image

import mindspore.dataset as ds
import mindspore.dataset.transforms.transforms as trans
import mindspore.dataset.vision.transforms as vision
from mindspore import log as logger
from util import diff_mse, visualize_list, save_and_check_md5_pil

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
TEST_DATA_DATASET_FUNC ="../data/dataset/"


def test_linear_transformation_op(plot=False):
    """
    Feature: LinearTransformation op
    Description: Test LinearTransformation op by verifying if images transform correctly
    Expectation: Output is equal to the expected output
    """
    logger.info("test_linear_transformation_01")

    # Initialize parameters
    height = 50
    weight = 50
    dim = 3 * height * weight
    transformation_matrix = np.eye(dim)
    mean_vector = np.zeros(dim)

    # Define operations
    transforms = [
        vision.Decode(True),
        vision.CenterCrop([height, weight]),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=transform, input_columns=["image"])
    # Note: if transformation matrix is diagonal matrix with all 1 in diagonal,
    #       the output matrix in expected to be the same as the input matrix.
    data1 = data1.map(operations=vision.LinearTransformation(transformation_matrix, mean_vector),
                      input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=transform, input_columns=["image"])

    image_transformed = []
    image = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image2 = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_transformed.append(image1)
        image.append(image2)

        mse = diff_mse(image1, image2)
        assert mse == 0
    if plot:
        visualize_list(image, image_transformed)


def test_linear_transformation_md5():
    """
    Feature: LinearTransformation op
    Description: Test LinearTransformation op with valid params (transformation_matrix, mean_vector) with md5 check
    Expectation: Pass the md5 check test
    """
    logger.info("test_linear_transformation_md5")

    # Initialize parameters
    height = 50
    weight = 50
    dim = 3 * height * weight
    transformation_matrix = np.ones([dim, dim])
    mean_vector = np.zeros(dim)

    # Generate dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        vision.Decode(True),
        vision.CenterCrop([height, weight]),
        vision.ToTensor(),
        vision.LinearTransformation(transformation_matrix, mean_vector)
    ]
    transform = trans.Compose(transforms)
    data1 = data1.map(operations=transform, input_columns=["image"])

    # Compare with expected md5 from images
    filename = "linear_transformation_01_result.npz"
    save_and_check_md5_pil(data1, filename, generate_golden=GENERATE_GOLDEN)


def test_linear_transformation_exception_01():
    """
    Feature: LinearTransformation op
    Description: Test LinearTransformation op when transformation_matrix is not provided
    Expectation: Error is raised as expected
    """
    logger.info("test_linear_transformation_exception_01")

    # Initialize parameters
    height = 50
    weight = 50
    dim = 3 * height * weight
    mean_vector = np.zeros(dim)

    # Generate dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    try:
        transforms = [
            vision.Decode(True),
            vision.CenterCrop([height, weight]),
            vision.ToTensor(),
            vision.LinearTransformation(None, mean_vector)
        ]
        transform = trans.Compose(transforms)
        data1 = data1.map(operations=transform, input_columns=["image"])
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Argument transformation_matrix with value None is not of type [<class 'numpy.ndarray'>]" in str(e)


def test_linear_transformation_exception_02():
    """
    Feature: LinearTransformation op
    Description: Test LinearTransformation op when mean_vector is not provided
    Expectation: Error is raised as expected
    """
    logger.info("test_linear_transformation_exception_02")

    # Initialize parameters
    height = 50
    weight = 50
    dim = 3 * height * weight
    transformation_matrix = np.ones([dim, dim])

    # Generate dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    try:
        transforms = [
            vision.Decode(True),
            vision.CenterCrop([height, weight]),
            vision.ToTensor(),
            vision.LinearTransformation(transformation_matrix, None)
        ]
        transform = trans.Compose(transforms)
        data1 = data1.map(operations=transform, input_columns=["image"])
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Argument mean_vector with value None is not of type [<class 'numpy.ndarray'>]" in str(e)


def test_linear_transformation_exception_03():
    """
    Feature: LinearTransformation op
    Description: Test LinearTransformation op when transformation_matrix is not a square matrix
    Expectation: Error is raised as expected
    """
    logger.info("test_linear_transformation_exception_03")

    # Initialize parameters
    height = 50
    weight = 50
    dim = 3 * height * weight
    transformation_matrix = np.ones([dim, dim - 1])
    mean_vector = np.zeros(dim)

    # Generate dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    try:
        transforms = [
            vision.Decode(True),
            vision.CenterCrop([height, weight]),
            vision.ToTensor(),
            vision.LinearTransformation(transformation_matrix, mean_vector)
        ]
        transform = trans.Compose(transforms)
        data1 = data1.map(operations=transform, input_columns=["image"])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "square matrix" in str(e)


def test_linear_transformation_exception_04():
    """
    Feature: LinearTransformation op
    Description: Test LinearTransformation op when mean_vector does not match dimension of transformation_matrix
    Expectation: Error is raised as expected
    """
    logger.info("test_linear_transformation_exception_04")

    # Initialize parameters
    height = 50
    weight = 50
    dim = 3 * height * weight
    transformation_matrix = np.ones([dim, dim])
    mean_vector = np.zeros(dim - 1)

    # Generate dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    try:
        transforms = [
            vision.Decode(True),
            vision.CenterCrop([height, weight]),
            vision.ToTensor(),
            vision.LinearTransformation(transformation_matrix, mean_vector)
        ]
        transform = trans.Compose(transforms)
        data1 = data1.map(operations=transform, input_columns=["image"])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "should match" in str(e)


def test_linear_transformation_operation_01():
    """
    Feature: LinearTransformation operation
    Description: Testing the normal functionality of the LinearTransformation operator
    Expectation: The Output is equal to the expected output
    """
    # LinearTransformation operator: Normal testing, Numpy Image shape = (3, 3, 5)
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    transformation_matrix = np.ones([45, 45])
    mean_vector = np.ones([45])
    dataset = ds.ImageFolderDataset(data_dir)
    transforms = [vision.Decode(),
                  vision.Resize(3),
                  vision.ToTensor(),
                  vision.LinearTransformation(transformation_matrix, mean_vector)]
    transform = trans.Compose(transforms)
    dataset = dataset.map(input_columns=["image"], operations=transform)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # LinearTransformation operator: Normal testing, Numpy Image shape = (3, 5, 8)
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    transformation_matrix = np.ones([120, 120])
    mean_vector = np.ones([120])
    dataset = ds.ImageFolderDataset(data_dir)
    transforms = [vision.Decode(),
                  vision.Resize(5),
                  vision.ToTensor(),
                  vision.LinearTransformation(transformation_matrix, mean_vector)]
    transform = trans.Compose(transforms)
    dataset = dataset.map(input_columns=["image"], operations=transform)

    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # LinearTransformation operator: Normal testing, Numpy Image shape = (3, 10, 17)
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    transformation_matrix = np.ones([510, 510])
    mean_vector = np.ones([510])
    dataset = ds.ImageFolderDataset(data_dir)
    transform = [vision.Decode(to_pil=True),
                 vision.Resize(10),
                 vision.ToTensor(),
                 vision.LinearTransformation(transformation_matrix, mean_vector)]
    transform = trans.Compose(transform)
    dataset = dataset.map(input_columns=["image"], operations=transform)

    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass


def test_linear_transformation_operation_02():
    """
    Feature: LinearTransformation operation
    Description: Testing the normal functionality of the LinearTransformation operator
    Expectation: The Output is equal to the expected output
    """
    # LinearTransformation operator: Normal testing, Numpy Image shape = (3, 16, 28)
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    transformation_matrix = np.ones([1344, 1344])
    mean_vector = np.ones([1344])
    dataset = ds.ImageFolderDataset(data_dir)
    transforms = [vision.Decode(to_pil=True),
                  vision.Resize(16),
                  vision.ToTensor(),
                  vision.LinearTransformation(transformation_matrix, mean_vector)]
    transform = trans.Compose(transforms)
    dataset = dataset.map(input_columns=["image"], operations=transform)

    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # LinearTransformation operator: Normal testing, Numpy Image shape = (3, 20, 35)
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    transformation_matrix = np.ones([2100, 2100])
    mean_vector = np.ones([2100])
    dataset = ds.ImageFolderDataset(data_dir)
    transforms = [vision.Decode(to_pil=True),
                  vision.Resize(20),
                  vision.ToTensor(),
                  vision.LinearTransformation(transformation_matrix, mean_vector)]
    transform = trans.Compose(transforms)
    dataset = dataset.map(input_columns=["image"], operations=transform)

    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # LinearTransformation operator: Normal testing, Numpy Image shape = (3, 25, 44)
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    transformation_matrix = np.ones([3300, 3300])
    mean_vector = np.ones([3300])
    dataset = ds.ImageFolderDataset(data_dir)
    transforms = [vision.Decode(),
                  vision.Resize(25),
                  vision.ToTensor(),
                  vision.LinearTransformation(transformation_matrix, mean_vector)]
    transform = trans.Compose(transforms)
    dataset = dataset.map(input_columns=["image"], operations=transform)

    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass


def test_linear_transformation_operation_03():
    """
    Feature: LinearTransformation operation
    Description: Testing the normal functionality of the LinearTransformation operator
    Expectation: The Output is equal to the expected output
    """
    # LinearTransformation operator: Normal testing, Numpy Image shape = (3, 32, 56)
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    transformation_matrix = np.ones([5376, 5376])
    mean_vector = np.ones([5376])
    dataset = ds.ImageFolderDataset(data_dir)
    transforms = [vision.Decode(to_pil=True),
                  vision.Resize(32),
                  vision.ToTensor(),
                  vision.LinearTransformation(transformation_matrix, mean_vector)]
    transform = trans.Compose(transforms)
    dataset = dataset.map(input_columns=["image"], operations=transform)

    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # LinearTransformation operator: Normal testing, transformation_matrix and mean_vector are numpy arrays.
    image = np.random.randn(10, 10, 3)
    transformation_matrix = np.random.randn(300, 300)
    mean_vector = np.random.randn(300,)
    linear_op = vision.LinearTransformation(transformation_matrix=transformation_matrix, mean_vector=mean_vector)
    _ = linear_op(image)

    # LinearTransformation operator: Normal testing, mean_vector is a numpy integer array
    image = np.random.randn(28, 32, 10)
    transformation_matrix = np.random.randn(8960, 8960)
    mean_vector = np.random.randint(-100, 100, (8960,))
    linear_op = vision.LinearTransformation(transformation_matrix=transformation_matrix, mean_vector=mean_vector)
    linear_op(image)

    # LinearTransformation operator: Normal testing, transformation_matrix is a numpy integer array
    image = np.random.randn(64, 5)
    transformation_matrix = np.random.randint(-255, 255, (320, 320))
    mean_vector = np.random.randint(-1024, 582, (320,))
    linear_op = vision.LinearTransformation(transformation_matrix=transformation_matrix, mean_vector=mean_vector)
    _ = linear_op(image)

    # LinearTransformation operator: Normal testing, Input data is a NumPy image.
    image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
    with Image.open(image_bmp) as image:
        image = vision.ToTensor()(image)
        image = np.transpose(image, (1, 2, 0))
        chw = np.array(image).shape[0] * np.array(image).shape[1] * np.array(image).shape[2]
        transformation_matrix = np.ones((chw, chw))
        mean_vector = np.ones((chw,))
        linear_op = vision.LinearTransformation(transformation_matrix=transformation_matrix, mean_vector=mean_vector)
        _ = linear_op(image)

    # LinearTransformation operator: Normal testing, input data为4维
    image = np.random.randn(20, 8, 10, 5)
    transformation_matrix = np.random.randint(-1000, 1000, (8000, 8000))
    mean_vector = np.random.randint(-1000, 1000, (8000,))
    linear_op = vision.LinearTransformation(transformation_matrix=transformation_matrix, mean_vector=mean_vector)
    linear_op(image)

    # LinearTransformation operator: Normal testing,Input data is 4-dimensional.
    image = np.random.randn(200,)
    transformation_matrix = np.random.randint(0, 1, (200, 200))
    mean_vector = np.random.randn(200,)
    linear_op = vision.LinearTransformation(transformation_matrix=transformation_matrix, mean_vector=mean_vector)
    linear_op(image)


def test_linear_transformation_exception_05():
    """
    Feature: LinearTransformation operation
    Description: Testing the LinearTransformation Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # LinearTransformation operator: Exception Testing, The transformation matrix is not (D, D)-dimensional.
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    transformation_matrix = np.ones([12, 10])
    mean_vector = np.ones([12])
    dataset = ds.ImageFolderDataset(data_dir)
    with pytest.raises(ValueError, match="transformation_matrix should be a square matrix"):
        transforms = [vision.Decode(),
                      vision.Resize(2),
                      vision.ToTensor(),
                      vision.LinearTransformation(transformation_matrix, mean_vector)]
        transform = trans.Compose(transforms)
        dataset = dataset.map(input_columns=["image"], operations=transform)
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # LinearTransformation operator: Exception Testing, The transformation matrix does not match the mean vector.
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    transformation_matrix = np.ones([12, 12])
    mean_vector = np.ones([10])
    dataset = ds.ImageFolderDataset(data_dir)
    with pytest.raises(ValueError, match=("mean_vector length 10 should match either one "
                                          "dimension of the squaretransformation_matrix")):
        transforms = [vision.Decode(),
                      vision.Resize(2),
                      vision.ToTensor(),
                      vision.LinearTransformation(transformation_matrix, mean_vector)]
        transform = trans.Compose(transforms)
        dataset = dataset.map(input_columns=["image"], operations=transform)

        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # LinearTransformation operator: Exception Testing, 缺失transformation_matrix
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    mean_vector = np.ones([12])
    dataset = ds.ImageFolderDataset(data_dir)
    with pytest.raises(TypeError, match="missing a required argument"):
        transforms = [vision.Decode(),
                      vision.Resize(2),
                      vision.ToTensor(),
                      vision.LinearTransformation(mean_vector=mean_vector)]
        transform = trans.Compose(transforms)
        dataset = dataset.map(input_columns=["image"], operations=transform)

        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass


def test_linear_transformation_exception_06():
    """
    Feature: LinearTransformation operation
    Description: Testing the LinearTransformation Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # LinearTransformation operator: Exception Testing, Missing mean_vector
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    transformation_matrix = np.ones([12, 12])
    dataset = ds.ImageFolderDataset(data_dir)
    with pytest.raises(TypeError, match="missing a required argument"):
        transforms = [vision.Decode(),
                      vision.Resize(2),
                      vision.ToTensor(),
                      vision.LinearTransformation(transformation_matrix=transformation_matrix)]
        transform = trans.Compose(transforms)
        dataset = dataset.map(input_columns=["image"], operations=transform)

        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # LinearTransformation operator: Exception Testing, No parameters passed
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir)
    with pytest.raises(TypeError, match="missing a required argument"):
        transforms = [vision.Decode(),
                      vision.Resize(2),
                      vision.ToTensor(),
                      vision.LinearTransformation()]
        transform = trans.Compose(transforms)
        dataset = dataset.map(input_columns=["image"], operations=transform)

        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # LinearTransformation operator: Exception Testing, Input data is a BMP image
    image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "test_cv_image", "bmp.bmp")
    with Image.open(image_bmp) as image:
        transformation_matrix = np.random.randint(0, 1, (50, 50))
        mean_vector = np.random.randn(50,)
        linear_op = vision.LinearTransformation(transformation_matrix=transformation_matrix, mean_vector=mean_vector)
        with pytest.raises(TypeError, match="img should be NumPy array. Got <class 'PIL.BmpImagePlugin.BmpImageFile'>"):
            linear_op(image)

    # LinearTransformation operator: Exception Testing, Input data is a list
    image = np.random.randn(200,).tolist()
    transformation_matrix = np.random.randn(200, 200)
    mean_vector = np.random.randn(200,)
    linear_op = vision.LinearTransformation(transformation_matrix=transformation_matrix, mean_vector=mean_vector)
    with pytest.raises(TypeError, match="img should be NumPy array. Got <class 'list'>"):
        linear_op(image)

    # LinearTransformation operator: Exception Testing, transformation_matrix do not match the input data.
    image = np.random.randn(200,)
    transformation_matrix = np.random.randn(199, 199)
    mean_vector = np.random.randn(199,)
    linear_op = vision.LinearTransformation(transformation_matrix=transformation_matrix, mean_vector=mean_vector)
    with pytest.raises(ValueError, match="transformation_matrix shape \\(199, 199\\) not compatible with "
                                         "Numpy image shape \\(200,\\)."):
        linear_op(image)

    # LinearTransformation operator: Exception Testing, transformation_matrix is a list
    transformation_matrix = np.random.randn(2, 2).tolist()
    mean_vector = np.random.randn(2,)
    with pytest.raises(TypeError, match="is not of type \\[<class 'numpy.ndarray'>\\]."):
        vision.LinearTransformation(transformation_matrix=transformation_matrix, mean_vector=mean_vector)

    # LinearTransformation operator: Exception Testing, transformation_matrix is a tuple
    transformation_matrix = tuple(np.random.randn(2, 2))
    mean_vector = np.random.randn(2,)
    with pytest.raises(TypeError, match="is not of type \\[<class 'numpy.ndarray'>\\]."):
        vision.LinearTransformation(transformation_matrix=transformation_matrix, mean_vector=mean_vector)


def test_linear_transformation_exception_07():
    """
    Feature: LinearTransformation operation
    Description: Testing the LinearTransformation Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # LinearTransformation operator: Exception Testing, mean_vector is a list
    transformation_matrix = np.random.randn(2, 2)
    mean_vector = np.random.randn(2,).tolist()
    with pytest.raises(TypeError, match="is not of type \\[<class 'numpy.ndarray'>\\]."):
        vision.LinearTransformation(transformation_matrix=transformation_matrix, mean_vector=mean_vector)

    # LinearTransformation operator: Exception Testing, mean_vector is a tuple
    transformation_matrix = np.random.randn(2, 2)
    mean_vector = tuple(np.random.randn(2,))
    with pytest.raises(TypeError, match="is not of type \\[<class 'numpy.ndarray'>\\]."):
        vision.LinearTransformation(transformation_matrix=transformation_matrix, mean_vector=mean_vector)

    # LinearTransformation operator: Exception Testing, The transformation_matrix is 3D.
    image = np.random.randn(10, 5)
    transformation_matrix = np.random.randn(50, 50, 50)
    mean_vector = np.random.randn(50,)
    linear_op = vision.LinearTransformation(transformation_matrix=transformation_matrix, mean_vector=mean_vector)
    with pytest.raises(ValueError):
        linear_op(image)

    # LinearTransformation operator: Exception Testing, The transformation_matrix has dimensions (D,)
    transformation_matrix = np.random.randn(50,)
    mean_vector = np.random.randn(50,)
    with pytest.raises(IndexError):
        vision.LinearTransformation(transformation_matrix=transformation_matrix, mean_vector=mean_vector)


if __name__ == '__main__':
    test_linear_transformation_op(plot=True)
    test_linear_transformation_md5()
    test_linear_transformation_operation_01()
    test_linear_transformation_operation_02()
    test_linear_transformation_operation_03()
    test_linear_transformation_exception_01()
    test_linear_transformation_exception_02()
    test_linear_transformation_exception_03()
    test_linear_transformation_exception_04()
    test_linear_transformation_exception_05()
    test_linear_transformation_exception_06()
    test_linear_transformation_exception_07()
