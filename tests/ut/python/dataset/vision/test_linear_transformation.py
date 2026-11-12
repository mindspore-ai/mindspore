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
import pytest

import mindspore.dataset.vision.transforms as vision

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
TEST_DATA_DATASET_FUNC ="../data/dataset/"
























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
