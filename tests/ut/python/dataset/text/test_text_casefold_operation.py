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
"""text transform - casefold"""

import os
import mindspore.dataset as ds
from mindspore.dataset import text


TEST_DATA_DATASET_FUNC ="../data/dataset/"

DATA_FILE = os.path.join(TEST_DATA_DATASET_FUNC,
                         "text_data/testTextFile/textfile/testTokenizerData/testCaseFold/english.txt")
DATA_FILE1 = os.path.join(TEST_DATA_DATASET_FUNC,
                          "text_data/testTextFile/textfile/testTokenizerData/testCaseFold/chinese.txt")
DATA_FILE2 = os.path.join(TEST_DATA_DATASET_FUNC,
                          "text_data/testTextFile/textfile/testTokenizerData/testCaseFold/cnanden.txt")
DATA_FILE3 = os.path.join(TEST_DATA_DATASET_FUNC,
                          "text_data/testTextFile/textfile/testTokenizerData/testCaseFold/numbers.txt")
DATA_FILE4 = os.path.join(TEST_DATA_DATASET_FUNC,
                          "text_data/testTextFile/textfile/testTokenizerData/testCaseFold/space.txt")
DATA_FILE5 = os.path.join(TEST_DATA_DATASET_FUNC,
                          "text_data/testTextFile/textfile/testTokenizerData/testCaseFold/punctuation.txt")
DATA_FILE6 = os.path.join(TEST_DATA_DATASET_FUNC,
                          "text_data/testTextFile/textfile/testTokenizerData/testCaseFold/enandnum.txt")
DATA_FILE7 = os.path.join(TEST_DATA_DATASET_FUNC,
                          "text_data/testTextFile/textfile/testTokenizerData/testCaseFold/mixed.txt")
DATA_FILE8 = "../data/dataset/testTokenizerData/1.txt"


def test_casefold_operation_01():
    """
    Feature: CaseFold op
    Description: Test CaseFold op with different character types (Chinese, English, numbers, symbols)
    Expectation: Successfully convert text to lowercase
    """
    # Test CaseFold, English and Chinese
    expect_strs = ["我喜欢english!"]
    dataset = ds.TextFileDataset(DATA_FILE2, shuffle=False)
    op = text.CaseFold()
    dataset = dataset.map(operations=op)
    lower_strs = []
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i["text"].tolist()
        lower_strs.append(data)
    assert lower_strs == expect_strs

    # Test CaseFold, numbers
    expect_strs = ["12345678"]
    dataset = ds.TextFileDataset(DATA_FILE3, shuffle=False)
    op = text.CaseFold()
    dataset = dataset.map(operations=op)

    lower_strs = []
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i["text"].tolist()
        lower_strs.append(data)
    assert lower_strs == expect_strs

    # Test CaseFold, space
    expect_strs = ["  "]
    dataset = ds.TextFileDataset(DATA_FILE4, shuffle=False)
    op = text.CaseFold()
    dataset = dataset.map(operations=op)

    lower_strs = []
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i["text"].tolist()
        lower_strs.append(data)
    assert lower_strs == expect_strs

    # Test CaseFold, punctuation
    expect_strs = ["#!#$^**&$$?><"]
    dataset = ds.TextFileDataset(DATA_FILE5, shuffle=False)
    op = text.CaseFold()
    dataset = dataset.map(operations=op)

    lower_strs = []
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i["text"].tolist()
        lower_strs.append(data)
    assert lower_strs == expect_strs

    # Test CaseFold, English and numbers
    expect_strs = ["hello world123!"]
    dataset = ds.TextFileDataset(DATA_FILE6, shuffle=False)
    op = text.CaseFold()
    dataset = dataset.map(operations=op)

    lower_strs = []
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i["text"].tolist()
        lower_strs.append(data)
    assert lower_strs == expect_strs

    # Test CaseFold, mixed
    expect_strs = ["welcome to beijing!", "北京欢迎您!", "我喜欢english!", "  "]
    dataset = ds.TextFileDataset(DATA_FILE7, shuffle=False)
    op = text.CaseFold()
    dataset = dataset.map(operations=op)

    lower_strs = []
    for i in dataset.create_dict_iterator(output_numpy=True):
        data = i["text"].tolist()
        lower_strs.append(data)
    assert lower_strs == expect_strs

    # Test CaseFold, data = "weLCome"
    data = "weLCome"
    op = text.CaseFold()
    data = op(data)
    assert data == "welcome"

    # Test CaseFold, data = "@#$%^@A"
    data = "@#$%^@A"
    op = text.CaseFold()
    data = op(data)
    assert data == "@#$%^@a"

    # Test CaseFold, data = "1234567B"
    data = "1234567B"
    op = text.CaseFold()
    data = op(data)
    assert data == "1234567b"

    # Test CaseFold, data = " "
    data = " "
    op = text.CaseFold()
    data = op(data)
    assert data == " "

    # Test CaseFold, data = "爱我中华"
    data = "爱我中华"
    op = text.CaseFold()
    data = op(data)
    assert data == "爱我中华"


def test_case_fold():
    """
    Feature: CaseFold op
    Description: Test CaseFold op basic usage
    Expectation: Output is equal to the expected output
    """
    expect_strs = ["welcome to beijing!", "北京欢迎您!", "我喜欢english!", "  "]
    dataset = ds.TextFileDataset(DATA_FILE8, shuffle=False)
    op = text.CaseFold()
    dataset = dataset.map(operations=op)

    lower_strs = []
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        token = i['text'].tolist()
        lower_strs.append(token)
    assert lower_strs == expect_strs
