# Copyright 2021 Huawei Technologies Co., Ltd
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

import mindspore.dataset as ds
import mindspore.dataset.audio.transforms as a_c_trans


def count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_expected) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}".format(
        data_expected[greater], data_me[greater], error[greater])


def test_func_angle_001():
    """
    Eager Test
    """
    arr = np.array([[73.04, -13.00], [57.49, 13.20], [-57.64, 6.51], [-52.25, 30.67], [-30.11, -18.34],
                    [-63.32, 99.33], [95.82, -24.76]], dtype=np.double)
    expected = np.array([-0.17614017, 0.22569334, 3.02912684, 2.6107975, -2.59450886, 2.13831337, -0.25286988],
                        dtype=np.double)
    angle_op = a_c_trans.Angle()
    output = angle_op(arr)
    count_unequal_element(expected, output, 0.0001, 0.0001)


def test_func_angle_002():
    """
    Pipeline Test
    """
    np.random.seed(6)
    arr = np.array([[[84.25, -85.92], [-92.23, 23.06], [-7.33, -44.17], [-62.95, -14.73]],
                    [[93.09, 38.18], [-81.94, 71.34], [71.33, -39.00], [95.25, -32.94]]], dtype=np.double)
    expected = np.array([[-0.79521156, 2.89658848, -1.73524737, -2.91173309],
                         [0.3892177, 2.42523905, -0.50034807, -0.33295219]], dtype=np.double)
    label = np.random.sample((2, 4, 1))
    data = (arr, label)
    dataset = ds.NumpySlicesDataset(data, column_names=["col1", "col2"], shuffle=False)
    angle_op = a_c_trans.Angle()
    dataset = dataset.map(operations=angle_op, input_columns=["col1"])
    for item1, item2 in zip(dataset.create_dict_iterator(output_numpy=True), expected):
        count_unequal_element(item2, item1['col1'], 0.0001, 0.0001)


def test_func_angle_003():
    """
    Pipeline Error Test
    """
    np.random.seed(78)
    arr = np.array([["11", "22"], ["33", "44"], ["55", "66"], ["77", "88"]])
    label = np.random.sample((4, 1))
    data = (arr, label)
    dataset = ds.NumpySlicesDataset(data, column_names=["col1", 'col2'], shuffle=False)
    angle_op = a_c_trans.Angle()
    dataset = dataset.map(operations=angle_op, input_columns=["col1"])
    num_itr = 0
    with pytest.raises(RuntimeError, match="input tensor type should be int, float or double"):
        for _ in dataset.create_dict_iterator(output_numpy=True):
            num_itr += 1


if __name__ == "__main__":
    test_func_angle_001()
    test_func_angle_002()
    test_func_angle_003()
