# Copyright 2021-2025 Huawei Technologies Co., Ltd
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
""" test audio transform - DCShift"""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from . import count_unequal_element


def test_dc_shift_eager():
    """
    Feature: DCShift
    Description: Test DCShift in eager mode
    Expectation: Output is equal to the expected output
    """
    arr = np.array([0.60, 0.97, -1.04, -1.26, 0.97, 0.91,
                    0.48, 0.93, 0.71, 0.61], dtype=np.double)
    expected = np.array([0.0400, 0.0400, -0.0400, -0.2600, 0.0400, 0.0400, 0.0400, 0.0400, 0.0400, 0.0400],
                        dtype=np.double)
    dc_shift = audio.DCShift(1.0, 0.04)
    output = dc_shift(arr)
    count_unequal_element(expected, output, 0.0001, 0.0001)


def test_dc_shift_pipeline():
    """
    Feature: DCShift
    Description: Test DCShift in pipeline mode
    Expectation: Output is equal to the expected output
    """
    arr = np.array([[1.14, -1.06, 0.94, 0.90],
                    [-1.11, 1.40, -0.33, 1.43]], dtype=np.double)
    expected = np.array([[0.2300, -0.2600, 0.2300, 0.2300],
                         [-0.3100, 0.2300, 0.4700, 0.2300]], dtype=np.double)
    dataset = ds.NumpySlicesDataset(arr, column_names=["col1"], shuffle=False)
    dc_shift = audio.DCShift(0.8, 0.03)
    dataset = dataset.map(operations=dc_shift, input_columns=["col1"])
    for item1, item2 in zip(dataset.create_dict_iterator(num_epochs=1, output_numpy=True), expected):
        count_unequal_element(item2, item1['col1'], 0.0001, 0.0001)


def test_dc_shift_pipeline_error():
    """
    Feature: DCShift
    Description: Test DCShift in pipeline mode with invalid input
    Expectation: Correct error and message are thrown as expected
    """
    arr = np.random.uniform(-2, 2, size=1000).astype(float)
    label = np.random.random_sample((1000, 1))
    data = (arr, label)
    dataset = ds.NumpySlicesDataset(
        data, column_names=["col1", "col2"], shuffle=False)
    num_itr = 0
    with pytest.raises(ValueError, match=r"Input shift is not within the required interval of \[-2.0, 2.0\]."):
        dc_shift = audio.DCShift(2.5, 0.03)
        dataset = dataset.map(operations=dc_shift, input_columns=["col1"])
        for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            num_itr += 1


if __name__ == "__main__":
    test_dc_shift_eager()
    test_dc_shift_pipeline()
    test_dc_shift_pipeline_error()
