# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
Testing unique op in DE
"""
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.transforms as ops


def compare(array, res, idx, cnt):
    data = ds.NumpySlicesDataset([array], column_names="x")
    data = data.batch(2)
    data = data.map(operations=ops.Unique(), input_columns=["x"], output_columns=["x", "y", "z"])
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        np.testing.assert_array_equal(res, d["x"])
        np.testing.assert_array_equal(idx, d["y"])
        np.testing.assert_array_equal(cnt, d["z"])

def test_duplicate_basics():
    """
    Feature: Unique op
    Description: Test Unique op basic usage where there are duplicates
    Expectation: Output is equal to the expected output
    """
    compare([0, 1, 2, 1, 2, 3], np.array([0, 1, 2, 3]),
            np.array([0, 1, 2, 1, 2, 3]), np.array([1, 2, 2, 1]))
    compare([0.0, 1.0, 2.0, 1.0, 2.0, 3.0], np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0, 1, 2, 1, 2, 3]), np.array([1, 2, 2, 1]))
    compare([1, 1, 1, 1, 1, 1], np.array([1]),
            np.array([0, 0, 0, 0, 0, 0]), np.array([6]))


if __name__ == "__main__":
    test_duplicate_basics()
