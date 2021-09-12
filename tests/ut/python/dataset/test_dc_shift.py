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


def test_func_dc_shift_eager():
    """
    Eager Test
    """
    arr = np.array([0.60, 0.97, -1.04, -1.26, 0.97, 0.91, 0.48, 0.93, 0.71, 0.61], dtype=np.double)
    expected = np.array([0.0400, 0.0400, -0.0400, -0.2600, 0.0400, 0.0400, 0.0400, 0.0400, 0.0400, 0.0400],
                        dtype=np.double)
    dcshift_op = a_c_trans.DCShift(1.0, 0.04)
    output = dcshift_op(arr)
    count_unequal_element(expected, output, 0.0001, 0.0001)


def test_func_dc_shift_pipeline():
    """
    Pipeline Test
    """
    arr = np.array([[1.14, -1.06, 0.94, 0.90], [-1.11, 1.40, -0.33, 1.43]], dtype=np.double)
    expected = np.array([[0.2300, -0.2600, 0.2300, 0.2300], [-0.3100, 0.2300, 0.4700, 0.2300]], dtype=np.double)
    dataset = ds.NumpySlicesDataset(arr, column_names=["col1"], shuffle=False)
    dcshift_op = a_c_trans.DCShift(0.8, 0.03)
    dataset = dataset.map(operations=dcshift_op, input_columns=["col1"])
    for item1, item2 in zip(dataset.create_dict_iterator(output_numpy=True), expected):
        count_unequal_element(item2, item1['col1'], 0.0001, 0.0001)


def test_func_dc_shift_pipeline_error():
    """
    Pipeline Error Test
    """
    arr = np.random.uniform(-2, 2, size=(1000)).astype(np.float)
    label = np.random.sample((1000, 1))
    data = (arr, label)
    dataset = ds.NumpySlicesDataset(data, column_names=["col1", "col2"], shuffle=False)
    num_itr = 0
    with pytest.raises(ValueError, match=r"Input shift is not within the required interval of \[-2.0, 2.0\]."):
        dcshift_op = a_c_trans.DCShift(2.5, 0.03)
        dataset = dataset.map(operations=dcshift_op, input_columns=["col1"])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            num_itr += 1


if __name__ == "__main__":
    test_func_dc_shift_eager()
    test_func_dc_shift_pipeline()
    test_func_dc_shift_pipeline_error()
