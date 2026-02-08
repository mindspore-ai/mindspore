# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""test text Glove"""

import numpy as np
import pytest

from mindspore import log
import mindspore.dataset as ds
from mindspore.dataset import text
import mindspore.dataset.text.transforms as T

DATASET_ROOT_PATH = "../data/dataset/testGloVe/"


def test_glove_all_build_from_file_params():
    """
    Feature: GloVe
    Description: Test with all parameters which include `path` and `max_vector` in function BuildFromFile
    Expectation: Output is equal to the expected value
    """
    vectors = text.GloVe.from_file(DATASET_ROOT_PATH + "glove.6B.test.txt", max_vectors=100)
    to_vectors = text.ToVectors(vectors)
    data = ds.TextFileDataset(DATASET_ROOT_PATH + "words.txt", shuffle=False)
    data = data.map(operations=to_vectors, input_columns=["text"])
    ind = 0
    res = [[0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411],
           [0, 0, 0, 0, 0, 0],
           [0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973],
           [0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603],
           [0.68047, -0.039263, 0.30186, -0.17792, 0.42962, 0.032246],
           [0.26818, 0.14346, -0.27877, 0.016257, 0.11384, 0.69923],
           [0, 0, 0, 0, 0, 0]]
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        res_array = np.array(res[ind], dtype=np.float32)
        assert np.array_equal(res_array, d["text"]), ind
        ind += 1


def test_glove_all_build_from_file_params_eager():
    """
    Feature: GloVe
    Description: Test with all parameters which include `path` and `max_vector` in function BuildFromFile in eager mode
    Expectation: Output is equal to the expected value
    """
    vectors = text.GloVe.from_file(DATASET_ROOT_PATH + "glove.6B.test.txt", max_vectors=4)
    to_vectors = T.ToVectors(vectors)
    result1 = to_vectors("ok")
    result2 = to_vectors("!")
    result3 = to_vectors("this")
    result4 = to_vectors("is")
    result5 = to_vectors("my")
    result6 = to_vectors("home")
    result7 = to_vectors("none")
    res = [[0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411],
           [0.013441, 0.23682, -0.16899, 0.40951, 0.63812, 0.47709],
           [0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973],
           [0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0]]
    res_array = np.array(res, dtype=np.float32)

    assert np.array_equal(result1, res_array[0])
    assert np.array_equal(result2, res_array[1])
    assert np.array_equal(result3, res_array[2])
    assert np.array_equal(result4, res_array[3])
    assert np.array_equal(result5, res_array[4])
    assert np.array_equal(result6, res_array[5])
    assert np.array_equal(result7, res_array[6])


def test_glove_all_to_vectors_params_eager():
    """
    Feature: GloVe
    Description: Test with all parameters which include `unk_init` and `lower_case_backup` in function ToVectors
        in eager mode
    Expectation: Output is equal to the expected value
    """
    vectors = text.GloVe.from_file(DATASET_ROOT_PATH + "glove.6B.test.txt", max_vectors=4)
    my_unk = [-1, -1, -1, -1, -1, -1]
    to_vectors = T.ToVectors(vectors, unk_init=my_unk, lower_case_backup=True)
    result1 = to_vectors("Ok")
    result2 = to_vectors("!")
    result3 = to_vectors("This")
    result4 = to_vectors("is")
    result5 = to_vectors("my")
    result6 = to_vectors("home")
    result7 = to_vectors("none")
    res = [[0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411],
           [0.013441, 0.23682, -0.16899, 0.40951, 0.63812, 0.47709],
           [0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973],
           [0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603],
           [-1, -1, -1, -1, -1, -1],
           [-1, -1, -1, -1, -1, -1],
           [-1, -1, -1, -1, -1, -1]]
    res_array = np.array(res, dtype=np.float32)

    assert np.array_equal(result1, res_array[0])
    assert np.array_equal(result2, res_array[1])
    assert np.array_equal(result3, res_array[2])
    assert np.array_equal(result4, res_array[3])
    assert np.array_equal(result5, res_array[4])
    assert np.array_equal(result6, res_array[5])
    assert np.array_equal(result7, res_array[6])


def test_glove_build_from_file():
    """
    Feature: GloVe
    Description: Test with only default parameter
    Expectation: Output is equal to the expected value
    """
    vectors = text.GloVe.from_file(DATASET_ROOT_PATH + "glove.6B.test.txt")
    to_vectors = text.ToVectors(vectors)
    data = ds.TextFileDataset(DATASET_ROOT_PATH + "words.txt", shuffle=False)
    data = data.map(operations=to_vectors, input_columns=["text"])
    ind = 0
    res = [[0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411],
           [0, 0, 0, 0, 0, 0],
           [0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973],
           [0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603],
           [0.68047, -0.039263, 0.30186, -0.17792, 0.42962, 0.032246],
           [0.26818, 0.14346, -0.27877, 0.016257, 0.11384, 0.69923],
           [0, 0, 0, 0, 0, 0]]
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        res_array = np.array(res[ind], dtype=np.float32)
        assert np.array_equal(res_array, d["text"]), ind
        ind += 1
    assert ind == 7


def test_glove_build_from_file_eager():
    """
    Feature: GloVe
    Description: Test with only default parameter in eager mode
    Expectation: Output is equal to the expected value
    """
    vectors = text.GloVe.from_file(DATASET_ROOT_PATH + "glove.6B.test.txt")
    to_vectors = T.ToVectors(vectors)
    result1 = to_vectors("ok")
    result2 = to_vectors("!")
    result3 = to_vectors("this")
    result4 = to_vectors("is")
    result5 = to_vectors("my")
    result6 = to_vectors("home")
    result7 = to_vectors("none")
    res = [[0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411],
           [0.013441, 0.23682, -0.16899, 0.40951, 0.63812, 0.47709],
           [0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973],
           [0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603],
           [0.68047, -0.039263, 0.30186, -0.17792, 0.42962, 0.032246],
           [0.26818, 0.14346, -0.27877, 0.016257, 0.11384, 0.69923],
           [0, 0, 0, 0, 0, 0]]
    res_array = np.array(res, dtype=np.float32)

    assert np.array_equal(result1, res_array[0])
    assert np.array_equal(result2, res_array[1])
    assert np.array_equal(result3, res_array[2])
    assert np.array_equal(result4, res_array[3])
    assert np.array_equal(result5, res_array[4])
    assert np.array_equal(result6, res_array[5])
    assert np.array_equal(result7, res_array[6])


def test_glove_invalid_input():
    """
    Feature: GloVe
    Description: Test the validate function with invalid parameters
    Expectation: Output is equal to the expected error
    """
    def test_invalid_input(test_name, file_path, error, error_msg, max_vectors=None, unk_init=None,
                           lower_case_backup=False, token="ok"):
        log.info("Test Vectors with wrong input: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            vectors = text.GloVe.from_file(file_path, max_vectors=max_vectors)
            to_vectors = T.ToVectors(vectors, unk_init=unk_init, lower_case_backup=lower_case_backup)
            to_vectors(token)
        assert error_msg in str(error_info.value)

    test_invalid_input("Not all vectors have the same number of dimensions",
                       DATASET_ROOT_PATH + "glove.6B.dim_different.txt", error=RuntimeError,
                       error_msg="all vectors must have the same number of dimensions, " \
                       "but got dim 5 while expecting 6")
    test_invalid_input("the file is empty.", DATASET_ROOT_PATH + "glove.6B.empty.txt",
                       error=RuntimeError, error_msg="invalid file, file is empty.")
    test_invalid_input("the count of `unknown_init`'s element is different with word vector.",
                       DATASET_ROOT_PATH + "glove.6B.test.txt",
                       error=RuntimeError,
                       error_msg="unk_init must be the same length as vectors, but got unk_init",
                       unk_init=[-1, -1])
    test_invalid_input("The file not exist", DATASET_ROOT_PATH + "not_exist.txt", RuntimeError,
                       error_msg="GloVe: invalid file")
    test_invalid_input("The token is 1-dimensional", DATASET_ROOT_PATH + "glove.6B.with_wrong_info.txt",
                       error=RuntimeError, error_msg="token with 1-dimensional vector.")
    test_invalid_input("max_vectors parameter must be greater than 0", DATASET_ROOT_PATH + "glove.6B.test.txt",
                       error=ValueError, error_msg="Input max_vectors is not within the required interval",
                       max_vectors=-1)
    test_invalid_input("invalid max_vectors parameter type as a float", DATASET_ROOT_PATH + "glove.6B.test.txt",
                       error=TypeError, error_msg="Argument max_vectors with value 1.0 is not of type [<class 'int'>],"
                       " but got <class 'float'>.", max_vectors=1.0)
    test_invalid_input("invalid max_vectors parameter type as a string", DATASET_ROOT_PATH + "glove.6B.test.txt",
                       error=TypeError, error_msg="Argument max_vectors with value 1 is not of type [<class 'int'>],"
                       " but got <class 'str'>.", max_vectors="1")
    test_invalid_input("invalid token parameter type as a float", DATASET_ROOT_PATH + "glove.6B.test.txt",
                       error=RuntimeError, error_msg="input tensor type should be string.", token=1.0)
    test_invalid_input("invalid lower_case_backup parameter type as a string", DATASET_ROOT_PATH + "glove.6B.test.txt",
                       error=TypeError, error_msg="Argument lower_case_backup with value True is " \
                       "not of type [<class 'bool'>],"
                       " but got <class 'str'>.", lower_case_backup="True")
    test_invalid_input("not right glove dataset. The formal must be `glove.6B.*.txt`", DATASET_ROOT_PATH +
                       "glove.6B.test.vec", error=RuntimeError, error_msg="GloVe: invalid file, can not " \
                       "find file 'glove.6B.*.txt'")


if __name__ == '__main__':
    test_glove_all_build_from_file_params()
    test_glove_all_build_from_file_params_eager()
    test_glove_all_to_vectors_params_eager()
    test_glove_build_from_file()
    test_glove_build_from_file_eager()
    test_glove_invalid_input()
