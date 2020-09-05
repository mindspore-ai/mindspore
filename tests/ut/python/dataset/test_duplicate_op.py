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
Testing Duplicate op in DE
"""
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as ops


def compare(array):
    data = ds.NumpySlicesDataset([array], column_names="x")
    array = np.array(array)
    data = data.map(operations=ops.Duplicate(), input_columns=["x"], output_columns=["x", "y"],
                    column_order=["x", "y"])
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        np.testing.assert_array_equal(array, d["x"])
        np.testing.assert_array_equal(array, d["y"])


def test_duplicate_basics():
    compare([1, 2, 3])
    compare([b"1", b"2", b"3"])


if __name__ == "__main__":
    test_duplicate_basics()
