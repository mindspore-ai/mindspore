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
"""text transform - charngram"""

import os
import numpy as np
import pytest
import mindspore.dataset as ds
from mindspore.dataset import text
import mindspore.dataset.text.transforms as T
from mindspore import log


TEST_DATA_DATASET_FUNC ="../data/dataset/"

vector_file1 = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testVectors", "char_n_gram_20_100d.txt")
vector_file2 = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testVectors", "char_n_gram_20_dim_different_100d.txt")
vector_file3 = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testVectors", "vector_test.txt")
vector_empty = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testVectors", "vectors_empty.txt")
text_file = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testVectors", "words.txt")

DATASET_ROOT_PATH = "../data/dataset/testVectors/"


def test_charngram_operation_01():
    """
    Feature: CharNGram op
    Description: Test CharNGram op with different max_vectors and unk_init parameters
    Expectation: Successfully load character n-gram vectors and convert tokens
    """
    # Description: test with only default parameter
    char_n_gram = text.CharNGram.from_file(vector_file1)
    a = []
    for _ in range(100):
        a.append(10)
    vectors_op = text.ToVectors(char_n_gram, a)
    data = ds.TextFileDataset(text_file, shuffle=False)
    data = data.map(operations=vectors_op, input_columns=["text"])
    ind = 0
    out = []
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        out.append(d["text"])
        ind += 1
    assert ind == 7
    assert (out[0] == a).all()

    # Description: test with only default parameter
    char_n_gram = text.CharNGram.from_file(vector_file1, 300)
    a = []
    for _ in range(100):
        a.append(10)
    vectors_op = text.ToVectors(char_n_gram, a)
    data = ds.TextFileDataset(text_file, shuffle=False)
    data = data.map(operations=vectors_op, input_columns=["text"])
    ind = 0
    out = []
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        out.append(d["text"])
        ind += 1
    assert ind == 7
    assert (out[0] == a).all()

    # Description: test with all parameters which include `path` and `max_vector` in function BuildFromFile in eager mode
    char_n_gram = text.CharNGram.from_file(vector_file1, max_vectors=18)
    vectors_op = text.ToVectors(char_n_gram)
    vectors_op("the")

    # test apply_func return ""
    def generator():
        text_list = ["te", "Ba", "ab", "D", "haha", "!", "%^", "1", "张"]
        for i in text_list:
            yield (np.array([i]),)

    char_n_gram = text.CharNGram.from_file(vector_file1, 10)
    a = []
    for i in range(100):
        a.append(i)
    vectors_op = text.ToVectors(char_n_gram, a)
    dataset = ds.GeneratorDataset(generator, ["text"], shuffle=False)
    dataset = dataset.map(operations=vectors_op, input_columns=["text"])
    numiter = 0
    for _ in dataset.create_dict_iterator(output_numpy=True):
        numiter += 1
    assert numiter == 9

    # test apply_func return ""
    def generator2():
        text_list = ["@#$", "l", " ", "//", "张"]
        for i in text_list:
            yield (np.array([i]),)

    char_n_gram = text.CharNGram.from_file(vector_file1, 100)
    a = []
    for i in range(100):
        a.append(i)
    vectors_op = text.ToVectors(char_n_gram, a)
    dataset = ds.GeneratorDataset(generator2, ["text"], shuffle=False)
    dataset = dataset.map(operations=vectors_op, input_columns=["text"])
    numiter = 0
    for _ in dataset.create_dict_iterator(output_numpy=True):
        numiter += 1
    assert numiter == 5


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


def test_charngram_exception_01():
    """
    Feature: CharNGram op
    Description: Test CharNGram op with invalid file formats and parameters
    Expectation: Raise expected exceptions for invalid vector files and parameters
    """
    # Description: test with only default parameter
    with pytest.raises(RuntimeError, match="Vectors: all vectors must have the same number"
                                           " of dimensions, but got dim 99 while expecting 100"):
        text.CharNGram.from_file(vector_file2)

    # vector is empty
    with pytest.raises(RuntimeError, match="Vectors: invalid file, file is empty."):
        text.CharNGram.from_file(vector_empty)

    # vector is empty
    with pytest.raises(TypeError, match="Argument file_path with value 1 is not of"
                                        " type \\[<class 'str'>\\], but got <class 'int'>."):
        text.CharNGram.from_file(1)

    # vector is empty
    with pytest.raises(RuntimeError, match="Vectors: "):
        text.CharNGram.from_file("no_text")

    # Description: test with only default parameter
    with pytest.raises(RuntimeError, match="ToVectors: unk_init must be the same length as"
                                           " vectors, but got unk_init: 10 and vectors: 100"):
        char_n_gram = text.CharNGram.from_file(vector_file1)
        a = []
        for _ in range(10):
            a.append(10)
        vectors_op = text.ToVectors(char_n_gram, a)
        data = ds.TextFileDataset(text_file, shuffle=False)
        data = data.map(operations=vectors_op, input_columns=["text"])
        for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass

    # vector is empty
    with pytest.raises(ValueError,
                       match="Input max_vectors is not within the required interval of \\[0, 2147483647\\]."):
        text.CharNGram.from_file(vector_file1, -1)

    # vector is empty
    with pytest.raises(ValueError,
                       match="Input max_vectors is not within the required interval of \\[0, 2147483647\\]."):
        text.CharNGram.from_file(vector_file1, 2147483648)

    # vector is empty
    with pytest.raises(TypeError, match="Argument max_vectors with value 10.0 is not of"
                                        " type \\[<class 'int'>\\], but got <class 'float'>."):
        text.CharNGram.from_file(vector_file1, 10.0)

    # vector is empty
    with pytest.raises(TypeError, match="Argument max_vectors with value True is not of"
                                        " type \\(<class 'int'>,\\), but got <class 'bool'>."):
        text.CharNGram.from_file(vector_file1, True)

    # vector is empty
    with pytest.raises(TypeError, match="missing a required argument: 'file_path'"):
        text.CharNGram.from_file()
