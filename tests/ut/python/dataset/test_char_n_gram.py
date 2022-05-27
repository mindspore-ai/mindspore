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

import numpy as np
import pytest

from mindspore import log
import mindspore.dataset as ds
import mindspore.dataset.text as text
import mindspore.dataset.text.transforms as T

DATASET_ROOT_PATH = "../data/dataset/testVectors/"


def _count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_expected)*rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count/total_count) < rtol,\
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}".\
        format(data_expected[greater], data_me[greater], error[greater])


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert True


def test_char_n_gram_all_to_vectors_params_eager():
    """
    Feature: CharNGram
    Description: Test with all parameters which include `unk_init`
        and `lower_case_backup` in function ToVectors in eager mode
    Expectation: Output is equal to the expected value
    """
    char_n_gram = text.CharNGram.from_file(DATASET_ROOT_PATH + "char_n_gram_20.txt", max_vectors=18)
    unk_init = (-np.ones(5)).tolist()
    to_vectors = T.ToVectors(char_n_gram, unk_init=unk_init, lower_case_backup=True)
    result1 = to_vectors("THE")
    result2 = to_vectors(".")
    result3 = to_vectors("To")
    res = [[-1.34121733e+00, 4.42693333e-02, -4.86969667e-01, 6.62939000e-01, -3.67669000e-01],
           [-1.00000000e+00, -1.00000000e+00, -1.00000000e+00, -1.00000000e+00, -1.00000000e+00],
           [-9.68530000e-01, -7.89463000e-01, 5.15762000e-01, 2.02107000e+00, -1.64635000e+00]]
    res_array = np.array(res, dtype=np.float32)

    allclose_nparray(res_array[0], result1, 0.0001, 0.0001)
    allclose_nparray(res_array[1], result2, 0.0001, 0.0001)
    allclose_nparray(res_array[2], result3, 0.0001, 0.0001)


def test_char_n_gram_build_from_file():
    """
    Feature: CharNGram
    Description: Test with only default parameter
    Expectation: Output is equal to the expected value
    """
    char_n_gram = text.CharNGram.from_file(DATASET_ROOT_PATH + "char_n_gram_20.txt")
    to_vectors = text.ToVectors(char_n_gram)
    data = ds.TextFileDataset(DATASET_ROOT_PATH + "words.txt", shuffle=False)
    data = data.map(operations=to_vectors, input_columns=["text"])
    ind = 0
    res = [[0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0.117336, 0.362446, -0.983326, 0.939264, -0.05648],
           [0.657201, 2.11761, -1.59276, 0.432072, 1.21395],
           [0., 0., 0., 0., 0.],
           [-2.26956, 0.288491, -0.740001, 0.661703, 0.147355],
           [0., 0., 0., 0., 0.]]
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        res_array = np.array(res[ind], dtype=np.float32)
        allclose_nparray(res_array, d["text"], 0.0001, 0.0001)
        ind += 1


def test_char_n_gram_all_build_from_file_params():
    """
    Feature: CharNGram
    Description: Test with all parameters which include `path` and `max_vector` in function BuildFromFile
    Expectation: Output is equal to the expected value
    """
    char_n_gram = text.CharNGram.from_file(DATASET_ROOT_PATH + "char_n_gram_20.txt", max_vectors=100)
    to_vectors = text.ToVectors(char_n_gram)
    data = ds.TextFileDataset(DATASET_ROOT_PATH + "words.txt", shuffle=False)
    data = data.map(operations=to_vectors, input_columns=["text"])
    ind = 0
    res = [[0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0.117336, 0.362446, -0.983326, 0.939264, -0.05648],
           [0.657201, 2.11761, -1.59276, 0.432072, 1.21395],
           [0., 0., 0., 0., 0.],
           [-2.26956, 0.288491, -0.740001, 0.661703, 0.147355],
           [0., 0., 0., 0., 0.]]
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        res_array = np.array(res[ind], dtype=np.float32)
        allclose_nparray(res_array, d["text"], 0.0001, 0.0001)
        ind += 1


def test_char_n_gram_all_build_from_file_params_eager():
    """
    Feature: CharNGram
    Description: Test with all parameters which include `path` and `max_vector` in function BuildFromFile in eager mode
    Expectation: Output is equal to the expected value
    """
    char_n_gram = text.CharNGram.from_file(DATASET_ROOT_PATH + "char_n_gram_20.txt", max_vectors=18)
    to_vectors = T.ToVectors(char_n_gram)
    result1 = to_vectors("the")
    result2 = to_vectors(".")
    result3 = to_vectors("to")
    res = [[-1.34121733e+00, 4.42693333e-02, -4.86969667e-01, 6.62939000e-01, -3.67669000e-01],
           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
           [-9.68530000e-01, -7.89463000e-01, 5.15762000e-01, 2.02107000e+00, -1.64635000e+00]]
    res_array = np.array(res, dtype=np.float32)

    allclose_nparray(res_array[0], result1, 0.0001, 0.0001)
    allclose_nparray(res_array[1], result2, 0.0001, 0.0001)
    allclose_nparray(res_array[2], result3, 0.0001, 0.0001)


def test_char_n_gram_build_from_file_eager():
    """
    Feature: CharNGram
    Description: Test with only default parameter in eager mode
    Expectation: Output is equal to the expected value
    """
    char_n_gram = text.CharNGram.from_file(DATASET_ROOT_PATH + "char_n_gram_20.txt")
    to_vectors = T.ToVectors(char_n_gram)
    result1 = to_vectors("the")
    result2 = to_vectors(".")
    result3 = to_vectors("to")
    res = [[-8.40079000e-01, -2.70002500e-02, -8.33472250e-01, 5.88367000e-01, -2.10011750e-01],
           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
           [-9.68530000e-01, -7.89463000e-01, 5.15762000e-01, 2.02107000e+00, -1.64635000e+00]]
    res_array = np.array(res, dtype=np.float32)

    allclose_nparray(res_array[0], result1, 0.0001, 0.0001)
    allclose_nparray(res_array[1], result2, 0.0001, 0.0001)
    allclose_nparray(res_array[2], result3, 0.0001, 0.0001)


def test_char_n_gram_invalid_input():
    """
    Feature: CharNGram
    Description: Test the validate function with invalid parameters.
    Expectation: Verification of correct error message for invalid input.
    """
    def test_invalid_input(test_name, file_path, error, error_msg, max_vectors=None,
                           unk_init=None, lower_case_backup=False, token="ok"):
        log.info("Test CharNGram with wrong input: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            char_n_gram = text.CharNGram.from_file(file_path, max_vectors=max_vectors)
            to_vectors = T.ToVectors(char_n_gram, unk_init=unk_init, lower_case_backup=lower_case_backup)
            to_vectors(token)
        assert error_msg in str(error_info.value)

    test_invalid_input("Not all vectors have the same number of dimensions",
                       DATASET_ROOT_PATH + "char_n_gram_20_dim_different.txt", error=RuntimeError,
                       error_msg="all vectors must have the same number of dimensions, " +
                       "but got dim 4 while expecting 5")
    test_invalid_input("the file is empty.", DATASET_ROOT_PATH + "vectors_empty.txt",
                       error=RuntimeError, error_msg="invalid file, file is empty.")
    test_invalid_input("the count of `unknown_init`'s element is different with word vector.",
                       DATASET_ROOT_PATH + "char_n_gram_20.txt",
                       error=RuntimeError, error_msg="unk_init must be the same length as vectors, " +
                       "but got unk_init: 6 and vectors: 5", unk_init=np.ones(6).tolist())
    test_invalid_input("The file not exist", DATASET_ROOT_PATH + "not_exist.txt", RuntimeError,
                       error_msg="get real path failed")
    test_invalid_input("max_vectors parameter must be greater than 0",
                       DATASET_ROOT_PATH + "char_n_gram_20.txt", error=ValueError,
                       error_msg="Input max_vectors is not within the required interval", max_vectors=-1)
    test_invalid_input("invalid max_vectors parameter type as a float",
                       DATASET_ROOT_PATH + "char_n_gram_20.txt", error=TypeError,
                       error_msg="Argument max_vectors with value 1.0 is not of type [<class 'int'>],"
                       " but got <class 'float'>.", max_vectors=1.0)
    test_invalid_input("invalid max_vectors parameter type as a string",
                       DATASET_ROOT_PATH + "char_n_gram_20.txt", error=TypeError,
                       error_msg="Argument max_vectors with value 1 is not of type [<class 'int'>],"
                       " but got <class 'str'>.", max_vectors="1")
    test_invalid_input("invalid token parameter type as a float",
                       DATASET_ROOT_PATH + "char_n_gram_20.txt", error=RuntimeError,
                       error_msg="input tensor type should be string.", token=1.0)
    test_invalid_input("invalid lower_case_backup parameter type as a string", DATASET_ROOT_PATH + "char_n_gram_20.txt",
                       error=TypeError, error_msg="Argument lower_case_backup with " +
                       "value True is not of type [<class 'bool'>],"
                       " but got <class 'str'>.", lower_case_backup="True")
    test_invalid_input("invalid lower_case_backup parameter type as a string", DATASET_ROOT_PATH + "char_n_gram_20.txt",
                       error=TypeError, error_msg="Argument lower_case_backup with " +
                       "value True is not of type [<class 'bool'>],"
                       " but got <class 'str'>.", lower_case_backup="True")


if __name__ == '__main__':
    test_char_n_gram_all_to_vectors_params_eager()
    test_char_n_gram_build_from_file()
    test_char_n_gram_all_build_from_file_params()
    test_char_n_gram_all_build_from_file_params_eager()
    test_char_n_gram_build_from_file_eager()
    test_char_n_gram_invalid_input()
