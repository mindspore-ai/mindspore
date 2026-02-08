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
"""text transform - truncate"""

from __future__ import absolute_import
import pytest
import mindspore.dataset as ds
from mindspore.dataset import text
from mindspore import log as logger


def test_truncate_operation_01():
    """
    Feature: Truncate op
    Description: Test Truncate op with different input types and max_seq_len values
    Expectation: Successfully truncate sequences to specified length
    """
    # truncate op：pipeline mode
    inputs = [['a', 'b', 'c', 'd', 'e']]
    dataset = ds.NumpySlicesDataset(data=inputs, column_names=["text"], shuffle=False)
    op = text.Truncate(4)
    dataset = dataset.map(operations=op, input_columns=["text"])
    for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        pass

    # truncate op：eager mode
    inputs = [['a', 'b', 'c', 'd', 'e']]
    op = text.Truncate(4)
    res = op(inputs)
    logger.info(res)

    # truncate op：input tuple
    inputs = (['a', 'b', 'c', 'd', 'e'])
    op = text.Truncate(4)
    res = op(inputs)
    logger.info(res)

    # truncate op：input list[int,]
    inputs = [1, 2, 3, 4, 5, 6]
    op = text.Truncate(4)
    res = op(inputs)
    logger.info(res)

    # truncate op：input rank is 1
    inputs = ['a', 'b', 'c', 'd', 'e', 'f']
    op = text.Truncate(4)
    res = op(inputs)
    logger.info(res)

    # truncate op：input rank is 2
    inputs = [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4], [5, 5, 5, 5, 5]]
    op = text.Truncate(4)
    res = op(inputs)
    logger.info(res)

    # truncate op：input rank is 3
    inputs = [[[1, 2, 3, 4, 5]]]
    op = text.Truncate(4)
    error_message = r"Truncate: the input tensor should be of dimension 1 or 2"
    with pytest.raises(RuntimeError, match=error_message):
        op(inputs)

    # truncate op：input max_seq_len=0
    inputs = [1, 2, 3, 4, 6, 6, 1]
    error_message = r"Input max_seq_len is not within the required interval of \[1, 2147483647\]."
    with pytest.raises(ValueError, match=error_message):
        op = text.Truncate(0)
        op(inputs)

    # truncate op：input max_seq_len=2147483647
    inputs = [1, 2, 3, 4, 6, 6, 1]
    op = text.Truncate(2147483647)
    res = op(inputs)
    logger.info(res)

    # truncate op：input max_seq_len<len(inputs)
    inputs = [1, 2, 3, 4, 6, 6, 1]
    op = text.Truncate(3)
    res = op(inputs)
    logger.info(res)


def test_truncate_exception_01():
    """
    Feature: Truncate op
    Description: Test Truncate op with invalid input types and out-of-range max_seq_len
    Expectation: Raise expected exceptions for invalid inputs
    """
    # truncate op：input str
    inputs = 'abcdef'
    op = text.Truncate(4)
    error_message = r"Truncate: the input tensor should be of dimension 1 or 2"
    with pytest.raises(RuntimeError, match=error_message):
        op(inputs)

    # truncate op：input int
    inputs = 456
    op = text.Truncate(4)
    error_message = r"Truncate: the input tensor should be of dimension 1 or 2"
    with pytest.raises(RuntimeError, match=error_message):
        op(inputs)

    # truncate op：input list[float,]
    inputs = [1., 2., 3., 4., 5., 6.]
    op = text.Truncate(4)
    res = op(inputs)
    assert (res == [1., 2., 3., 4.]).all()

    # truncate op：input max_seq_len=-1
    inputs = [1, 2, 3, 4, 6, 6, 1]

    error_message = r"Input max_seq_len is not within the required interval of \[1, 2147483647\]."
    with pytest.raises(ValueError, match=error_message):
        op = text.Truncate(-1)
        op(inputs)

    # truncate op：input max_seq_len=2147483648
    inputs = [1, 2, 3, 4, 6, 6, 1]

    error_message = r"Input max_seq_len is not within the required interval of \[1, 2147483647\]."
    with pytest.raises(ValueError, match=error_message):
        op = text.Truncate(2147483648)
        op(inputs)

    # truncate op：input max_seq_len str
    inputs = [1, 2, 3, 4, 6, 6, 1]

    error_message = r"Argument max_seq_len with value 12 is not of type \[<class 'int'>\], but got <class 'str'>"
    with pytest.raises(TypeError, match=error_message):
        op = text.Truncate('12')
        op(inputs)

    # truncate op：input max_seq_len float
    inputs = [1, 2, 3, 4, 6, 6, 1]

    error_message = r"Argument max_seq_len with value 1.5 is not of type \[<class 'int'>\], but got <class 'float'>."
    with pytest.raises(TypeError, match=error_message):
        op = text.Truncate(1.5)
        op(inputs)
