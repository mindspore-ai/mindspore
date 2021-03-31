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
Testing LinearTransformation op in DE
"""
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.transforms.py_transforms
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore import log as logger
from util import diff_mse, visualize_list, save_and_check_md5

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_linear_transformation_op(plot=False):
    """
    Test LinearTransformation op: verify if images transform correctly
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
        py_vision.Decode(),
        py_vision.CenterCrop([height, weight]),
        py_vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=transform, input_columns=["image"])
    # Note: if transformation matrix is diagonal matrix with all 1 in diagonal,
    #       the output matrix in expected to be the same as the input matrix.
    data1 = data1.map(operations=py_vision.LinearTransformation(transformation_matrix, mean_vector),
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
    Test LinearTransformation op: valid params (transformation_matrix, mean_vector)
    Expected to pass
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
        py_vision.Decode(),
        py_vision.CenterCrop([height, weight]),
        py_vision.ToTensor(),
        py_vision.LinearTransformation(transformation_matrix, mean_vector)
    ]
    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
    data1 = data1.map(operations=transform, input_columns=["image"])

    # Compare with expected md5 from images
    filename = "linear_transformation_01_result.npz"
    save_and_check_md5(data1, filename, generate_golden=GENERATE_GOLDEN)


def test_linear_transformation_exception_01():
    """
    Test LinearTransformation op: transformation_matrix is not provided
    Expected to raise ValueError
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
            py_vision.Decode(),
            py_vision.CenterCrop([height, weight]),
            py_vision.ToTensor(),
            py_vision.LinearTransformation(None, mean_vector)
        ]
        transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
        data1 = data1.map(operations=transform, input_columns=["image"])
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Argument transformation_matrix with value None is not of type [<class 'numpy.ndarray'>]" in str(e)


def test_linear_transformation_exception_02():
    """
    Test LinearTransformation op: mean_vector is not provided
    Expected to raise ValueError
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
            py_vision.Decode(),
            py_vision.CenterCrop([height, weight]),
            py_vision.ToTensor(),
            py_vision.LinearTransformation(transformation_matrix, None)
        ]
        transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
        data1 = data1.map(operations=transform, input_columns=["image"])
    except TypeError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Argument mean_vector with value None is not of type [<class 'numpy.ndarray'>]" in str(e)


def test_linear_transformation_exception_03():
    """
    Test LinearTransformation op: transformation_matrix is not a square matrix
    Expected to raise ValueError
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
            py_vision.Decode(),
            py_vision.CenterCrop([height, weight]),
            py_vision.ToTensor(),
            py_vision.LinearTransformation(transformation_matrix, mean_vector)
        ]
        transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
        data1 = data1.map(operations=transform, input_columns=["image"])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "square matrix" in str(e)


def test_linear_transformation_exception_04():
    """
    Test LinearTransformation op: mean_vector does not match dimension of transformation_matrix
    Expected to raise ValueError
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
            py_vision.Decode(),
            py_vision.CenterCrop([height, weight]),
            py_vision.ToTensor(),
            py_vision.LinearTransformation(transformation_matrix, mean_vector)
        ]
        transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
        data1 = data1.map(operations=transform, input_columns=["image"])
    except ValueError as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "should match" in str(e)


if __name__ == '__main__':
    test_linear_transformation_op(plot=True)
    test_linear_transformation_md5()
    test_linear_transformation_exception_01()
    test_linear_transformation_exception_02()
    test_linear_transformation_exception_03()
    test_linear_transformation_exception_04()
