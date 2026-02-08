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
"""text transform - slidingwindow"""

import numpy as np
import pytest
import mindspore.dataset as ds
from mindspore.dataset import text


def test_slidingwindow_operation_01():
    """
    Feature: SlidingWindow op
    Description: Test SlidingWindow op with different width parameters
    Expectation: Output matches expected sliding window results
    """
    # Test width  2
    inputs = ["happy", "birthday", "to", "you"]
    op = text.SlidingWindow(2, 0)
    res = op(inputs)
    assert np.array_equal(res, [['happy', 'birthday'], ['birthday', 'to'], ['to', 'you']])

    # Test width  4
    inputs = ["happy", "birthday", "to", "you"]
    op = text.SlidingWindow(4)
    res = op(inputs)
    assert np.array_equal(res, [['happy', 'birthday', 'to', 'you']])


def test_slidingwindow_exception_01():
    """
    Feature: SlidingWindow op
    Description: Test SlidingWindow op with invalid parameters
    Expectation: Raise expected exceptions for invalid inputs
    """
    # Test width = 0
    inputs = [["happy", "birthday", "to", "you"]]
    dataset = ds.NumpySlicesDataset(inputs, column_names=["text"], shuffle=False)
    with pytest.raises(ValueError, match=r"Input width is not within the required interval of \[1, 2147483647\]"):
        dataset.map(operations=text.SlidingWindow(0, 0), input_columns=["text"])

    # Test width = "2"
    inputs = [["happy", "birthday", "to", "you"]]

    dataset = ds.NumpySlicesDataset(inputs, column_names=["text"], shuffle=False)
    with pytest.raises(TypeError, match=r"Argument width with value 2 is not of type \[\<class \'int\'\>\]."):
        dataset.map(operations=text.SlidingWindow("2", 0), input_columns=["text"])

    # Test axis = 2
    inputs = [["happy", "birthday", "to", "you"]]
    expect = np.array([['happy', 'birthday'], ['birthday', 'to'], ['to', 'you']])

    dataset = ds.NumpySlicesDataset(inputs, column_names=["text"], shuffle=False)
    dataset = dataset.map(operations=text.SlidingWindow(2, 2), input_columns=["text"])
    with pytest.raises(RuntimeError, match="axis supports 0 or -1 only for now"):
        for data in dataset.create_dict_iterator(output_numpy=True):
            np.testing.assert_array_equal(data['number'], expect)

    # Test axis = "0"
    inputs = [["happy", "birthday", "to", "you"]]

    dataset = ds.NumpySlicesDataset(inputs, column_names=["text"], shuffle=False)
    with pytest.raises(TypeError, match=r"Argument axis with value 0 is not of type \[\<class \'int\'\>\]."):
        dataset.map(operations=text.SlidingWindow(2, "0"), input_columns=["text"])

    # Test inputs = ["aa", "bb", "cc"]
    inputs = ["aa", "bb", "cc"]

    dataset = ds.NumpySlicesDataset(inputs, column_names=["text"], shuffle=False)
    dataset = dataset.map(operations=text.SlidingWindow(2, 0), input_columns=["text"])

    result = []
    with pytest.raises(RuntimeError, match="SlidingWindow supports 1D input only for now"):
        for data in dataset.create_dict_iterator(output_numpy=True):
            for i in range(data['text'].shape[0]):
                result.append([])
                for j in range(data['text'].shape[1]):
                    result[i].append(data['text'][i][j].decode('utf8'))
            result = np.array(result)

    # Test no width
    inputs = [["happy", "birthday", "to", "you"]]

    dataset = ds.NumpySlicesDataset(inputs, column_names=["text"], shuffle=False)
    with pytest.raises(TypeError, match="missing a required argument: 'width'"):
        dataset.map(operations=text.SlidingWindow(axis=0), input_columns=["text"])

    # Test inputs = ''
    inputs = ''
    op = text.SlidingWindow(4)
    with pytest.raises(RuntimeError, match='SlidingWindow supports 1D input only for now.'):
        _ = op(inputs)

    # Test width = (2,1)
    with pytest.raises(TypeError, match=r'Argument width with value \(2, 1\) is not of type \[\<class \'int\'\>\].'):
        _ = text.SlidingWindow((2, 1))
