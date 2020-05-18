# Copyright 2019 Huawei Technologies Co., Ltd
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
import mindspore._c_dataengine as cde
import mindspore.dataset as ds
import pytest

import numpy as np


def test_basic():
    x = np.array([["ab", "cde", "121"], ["x", "km", "789"]], dtype='S')
    # x = np.array(["ab", "cde"], dtype='S')
    n = cde.Tensor(x)
    arr = n.as_array()
    y = np.array([1, 2])
    assert all(y == y)
    # assert np.testing.assert_array_equal(y,y)


def compare(strings):
    arr = np.array(strings, dtype='S')

    def gen():
        yield arr,

    data = ds.GeneratorDataset(gen, column_names=["col"])

    for d in data:
        np.testing.assert_array_equal(d[0], arr)


def test_generator():
    compare(["ab"])
    compare(["ab", "cde", "121"])
    compare([["ab", "cde", "121"], ["x", "km", "789"]])


def test_batching_strings():
    def gen():
        yield np.array(["ab", "cde", "121"], dtype='S'),

    data = ds.GeneratorDataset(gen, column_names=["col"]).batch(10)

    with pytest.raises(RuntimeError) as info:
        for _ in data:
            pass
    assert "[Batch ERROR] Batch does not support" in str(info)


if __name__ == '__main__':
    test_generator()
    test_basic()
    test_batching_strings()
