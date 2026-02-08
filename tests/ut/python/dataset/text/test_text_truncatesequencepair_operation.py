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
"""text transform - unicodechartokenizer"""

import numpy as np
import pytest
import mindspore.dataset as ds
from mindspore.dataset import text


def test_truncatesequencepair_operation_01():
    """
    Feature: TruncateSequencePair op
    Description: Test TruncateSequencePair op with different sequence lengths
    Expectation: Successfully truncate two sequences to specified total length
    """
    # test length = 4
    in1 = [1, 2]
    in2 = [4, 5]
    length = 4
    data = ds.NumpySlicesDataset({"s1": [in1], "s2": [in2]})
    data = data.map(operations=text.TruncateSequencePair(length), input_columns=["s1", "s2"])
    out1 = [1, 2]
    out2 = [4, 5]
    for d in data.create_dict_iterator(output_numpy=True):
        np.testing.assert_array_equal(out1, d["s1"])
        np.testing.assert_array_equal(out2, d["s2"])

    # test length = 4
    in1 = [1, 2, 3, 4]
    in2 = [5]
    length = 4
    data = ds.NumpySlicesDataset({"s1": [in1], "s2": [in2]})
    data = data.map(operations=text.TruncateSequencePair(length), input_columns=["s1", "s2"])
    out1 = [1, 2, 3]
    out2 = [5]
    for d in data.create_dict_iterator(output_numpy=True):
        np.testing.assert_array_equal(out1, d["s1"])
        np.testing.assert_array_equal(out2, d["s2"])

    # test length = 4
    in1 = [1, 2, 3, 4]
    in2 = [5, 6, 7, 8]
    length = 4
    data = ds.NumpySlicesDataset({"s1": [in1], "s2": [in2]})
    data = data.map(operations=text.TruncateSequencePair(length), input_columns=["s1", "s2"])
    out1 = [1, 2]
    out2 = [5, 6]
    for d in data.create_dict_iterator(output_numpy=True):
        np.testing.assert_array_equal(out1, d["s1"])
        np.testing.assert_array_equal(out2, d["s2"])

    # test length = 3
    in1 = [1, 2, 3]
    in2 = [4, 5]
    length = 3
    data = ds.NumpySlicesDataset({"s1": [in1], "s2": [in2]})
    data = data.map(operations=text.TruncateSequencePair(length), input_columns=["s1", "s2"])
    out1 = [1, 2]
    out2 = [4]
    for d in data.create_dict_iterator(output_numpy=True):
        np.testing.assert_array_equal(out1, d["s1"])
        np.testing.assert_array_equal(out2, d["s2"])

    # test length = 3
    in1 = [1, 2]
    in2 = [4, 5]
    length = 3
    data = ds.NumpySlicesDataset({"s1": [in1], "s2": [in2]})
    data = data.map(operations=text.TruncateSequencePair(length), input_columns=["s1", "s2"])
    out1 = [1, 2]
    out2 = [4]
    for d in data.create_dict_iterator(output_numpy=True):
        np.testing.assert_array_equal(out1, d["s1"])
        np.testing.assert_array_equal(out2, d["s2"])

    # test length = 5
    in1 = [1]
    in2 = [4]
    length = 5
    data = ds.NumpySlicesDataset({"s1": [in1], "s2": [in2]})
    data = data.map(operations=text.TruncateSequencePair(length), input_columns=["s1", "s2"])
    out1 = [1]
    out2 = [4]
    for d in data.create_dict_iterator(output_numpy=True):
        np.testing.assert_array_equal(out1, d["s1"])
        np.testing.assert_array_equal(out2, d["s2"])

    # test length = 3
    in1 = [1, 2, 3, 4]
    in2 = [5]
    length = 3
    data = ds.NumpySlicesDataset({"s1": [in1], "s2": [in2]})
    data = data.map(operations=text.TruncateSequencePair(length), input_columns=["s1", "s2"])
    out1 = [1, 2]
    out2 = [5]
    for d in data.create_dict_iterator(output_numpy=True):
        np.testing.assert_array_equal(out1, d["s1"])
        np.testing.assert_array_equal(out2, d["s2"])


def test_truncatesequencepair_operation_02():
    """
    Feature: TruncateSequencePair op
    Description: Test TruncateSequencePair op with different data types (bytes, int)
    Expectation: Successfully truncate sequences with mixed data types
    """
    # test length = 3
    in1 = [1, 2, 3, 4]
    in2 = [5, 6, 7, 8]
    length = 3
    data = ds.NumpySlicesDataset({"s1": [in1], "s2": [in2]})
    data = data.map(operations=text.TruncateSequencePair(length), input_columns=["s1", "s2"])
    out1 = [1, 2]
    out2 = [5]
    for d in data.create_dict_iterator(output_numpy=True):
        np.testing.assert_array_equal(out1, d["s1"])
        np.testing.assert_array_equal(out2, d["s2"])

    # test length = 4
    in1 = [b"1", b"2", b"3"]
    in2 = [4, 5]
    length = 4
    data = ds.NumpySlicesDataset({"s1": [in1], "s2": [in2]})
    data = data.map(operations=text.TruncateSequencePair(length), input_columns=["s1", "s2"])
    out1 = [b"1", b"2"]
    out2 = [4, 5]
    for d in data.create_dict_iterator(output_numpy=True):
        np.testing.assert_array_equal(out1, d["s1"])
        np.testing.assert_array_equal(out2, d["s2"])

    # test length = 4
    in1 = [b"1", b"2"]
    in2 = [b"4", b"5"]
    length = 4
    data = ds.NumpySlicesDataset({"s1": [in1], "s2": [in2]})
    data = data.map(operations=text.TruncateSequencePair(length), input_columns=["s1", "s2"])
    out1 = [b"1", b"2"]
    out2 = [b"4", b"5"]
    for d in data.create_dict_iterator(output_numpy=True):
        np.testing.assert_array_equal(out1, d["s1"])
        np.testing.assert_array_equal(out2, d["s2"])

    # test length = 4
    in1 = [b"1"]
    in2 = [4]
    length = 4
    data = ds.NumpySlicesDataset({"s1": [in1], "s2": [in2]})
    data = data.map(operations=text.TruncateSequencePair(length), input_columns=["s1", "s2"])
    out1 = [b"1"]
    out2 = [4]
    for d in data.create_dict_iterator(output_numpy=True):
        np.testing.assert_array_equal(out1, d["s1"])
        np.testing.assert_array_equal(out2, d["s2"])

    # test length = 4
    in1 = [b"1", b"2", b"3", b"4"]
    in2 = [b"5"]
    length = 4
    data = ds.NumpySlicesDataset({"s1": [in1], "s2": [in2]})
    data = data.map(operations=text.TruncateSequencePair(length), input_columns=["s1", "s2"])
    out1 = [b"1", b"2", b"3"]
    out2 = [b"5"]
    for d in data.create_dict_iterator(output_numpy=True):
        np.testing.assert_array_equal(out1, d["s1"])
        np.testing.assert_array_equal(out2, d["s2"])

    # test length = 4
    in1 = [b"1", b"2", b"3", b"4"]
    in2 = [5, 6, 7, 8]
    length = 4
    data = ds.NumpySlicesDataset({"s1": [in1], "s2": [in2]})
    data = data.map(operations=text.TruncateSequencePair(length), input_columns=["s1", "s2"])
    out1 = [b"1", b"2"]
    out2 = [5, 6]
    for d in data.create_dict_iterator(output_numpy=True):
        np.testing.assert_array_equal(out1, d["s1"])
        np.testing.assert_array_equal(out2, d["s2"])

    # test length = 4
    data = [["1", "2", "3"], ["4", "5"]]
    op = text.TruncateSequencePair(4)
    result = op(*data)
    assert np.array_equal(result[0], ['1', '2'])
    assert np.array_equal(result[1], ['4', '5'])

    # test length = 5
    data = [["1", "2", "3"], ["4", "5", "7", "8"]]
    op = text.TruncateSequencePair(5)
    result = op(*data)
    assert np.array_equal(result[0], ['1', '2', '3'])
    assert np.array_equal(result[1], ['4', '5'])

    # test length = 3
    data = [["1", "2", "3"], ["4", "5", "7", "8"]]
    op = text.TruncateSequencePair(3)
    result = op(*data)
    assert np.array_equal(result[0], ['1', '2'])
    assert np.array_equal(result[1], ['4'])


def test_truncatesequencepair_operation_03():
    """
    Feature: TruncateSequencePair op
    Description: Test TruncateSequencePair op in eager mode with tuple inputs
    Expectation: Successfully truncate tuple sequences and handle edge cases
    """
    # test length = 6
    data = (("he", "ll", "o"), ("wo", "r", "ld"))
    op = text.TruncateSequencePair(6)
    result = op(*data)
    assert np.array_equal(result[0], ['he', 'll', 'o'])
    assert np.array_equal(result[1], ['wo', 'r', 'ld'])

    # test length = 6  data = [["he", "ll", "o"], ["wo", "r", "ld"], ['1','2']]
    data = [["he", "ll", "o"], ["wo", "r", "ld"], ['1', '2']]
    op = text.TruncateSequencePair(6)
    with pytest.raises(RuntimeError, match='TruncateSequencePair: Expected two inputs.'):
        _ = op(*data)
