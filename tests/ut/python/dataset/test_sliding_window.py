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
Testing SlidingWindow in mindspore.dataset
"""
import numpy as np
import pytest
import mindspore.dataset as ds
import mindspore.dataset.text as text


def test_sliding_window_callable():
    """
    Test sliding window op is callable
    """
    op = text.SlidingWindow(2, 0)

    input1 = ["大", "家", "早", "上", "好"]
    expect = np.array([['大', '家'], ['家', '早'], ['早', '上'], ['上', '好']])
    result = op(input1)
    assert np.array_equal(result, expect)

    # test 2D input
    input2 = [["大", "家", "早", "上", "好"]]
    with pytest.raises(RuntimeError) as info:
        _ = op(input2)
    assert "SlidingWindow: SlidingWindow supports 1D input only for now." in str(info.value)

    # test input multiple tensors
    with pytest.raises(RuntimeError) as info:
        _ = op(input1, input1)
    assert "The op is OneToOne, can only accept one tensor as input." in str(info.value)


def test_sliding_window_string():
    """ test sliding_window with string type"""
    inputs = [["大", "家", "早", "上", "好"]]
    expect = np.array([['大', '家'], ['家', '早'], ['早', '上'], ['上', '好']])

    dataset = ds.NumpySlicesDataset(inputs, column_names=["text"], shuffle=False)
    dataset = dataset.map(operations=text.SlidingWindow(2, 0), input_columns=["text"])

    result = []
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        for i in range(data['text'].shape[0]):
            result.append([])
            for j in range(data['text'].shape[1]):
                result[i].append(data['text'][i][j].decode('utf8'))
        result = np.array(result)
    np.testing.assert_array_equal(result, expect)


def test_sliding_window_number():
    inputs = [1]
    expect = np.array([[1]])

    def gen(nums):
        yield (np.array(nums),)

    dataset = ds.GeneratorDataset(gen(inputs), column_names=["number"])
    dataset = dataset.map(operations=text.SlidingWindow(1, -1), input_columns=["number"])

    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        np.testing.assert_array_equal(data['number'], expect)


def test_sliding_window_big_width():
    inputs = [[1, 2, 3, 4, 5]]
    expect = np.array([])

    dataset = ds.NumpySlicesDataset(inputs, column_names=["number"], shuffle=False)
    dataset = dataset.map(operations=text.SlidingWindow(30, 0), input_columns=["number"])

    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        np.testing.assert_array_equal(data['number'], expect)


def test_sliding_window_exception():
    try:
        _ = text.SlidingWindow(0, 0)
        assert False
    except ValueError:
        pass

    try:
        _ = text.SlidingWindow("1", 0)
        assert False
    except TypeError:
        pass

    try:
        _ = text.SlidingWindow(1, "0")
        assert False
    except TypeError:
        pass

    try:
        inputs = [[1, 2, 3, 4, 5]]
        dataset = ds.NumpySlicesDataset(inputs, column_names=["text"], shuffle=False)
        dataset = dataset.map(operations=text.SlidingWindow(3, -100), input_columns=["text"])
        for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass
        assert False
    except RuntimeError as e:
        assert "axis supports 0 or -1 only for now." in str(e)

    try:
        inputs = ["aa", "bb", "cc"]
        dataset = ds.NumpySlicesDataset(inputs, column_names=["text"], shuffle=False)
        dataset = dataset.map(operations=text.SlidingWindow(2, 0), input_columns=["text"])
        for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass
        assert False
    except RuntimeError as e:
        assert "SlidingWindow supports 1D input only for now." in str(e)


if __name__ == '__main__':
    test_sliding_window_callable()
    test_sliding_window_string()
    test_sliding_window_number()
    test_sliding_window_big_width()
    test_sliding_window_exception()
