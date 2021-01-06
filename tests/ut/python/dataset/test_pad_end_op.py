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
Testing PadEnd op in DE
"""
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as ops


# Extensive testing of PadEnd is already done in batch with Pad test cases

def pad_compare(array, pad_shape, pad_value, res):
    data = ds.NumpySlicesDataset([array])
    if pad_value is not None:
        data = data.map(operations=ops.PadEnd(pad_shape, pad_value))
    else:
        data = data.map(operations=ops.PadEnd(pad_shape))
    for d in data.create_tuple_iterator(output_numpy=True):
        np.testing.assert_array_equal(res, d[0])


def test_pad_end_basics():
    pad_compare([1, 2], [3], -1, [1, 2, -1])
    pad_compare([1, 2, 3], [3], -1, [1, 2, 3])
    pad_compare([1, 2, 3], [2], -1, [1, 2])
    pad_compare([1, 2, 3], [5], None, [1, 2, 3, 0, 0])


def test_pad_end_str():
    pad_compare([b"1", b"2"], [3], b"-1", [b"1", b"2", b"-1"])
    pad_compare([b"1", b"2", b"3"], [3], b"-1", [b"1", b"2", b"3"])
    pad_compare([b"1", b"2", b"3"], [2], b"-1", [b"1", b"2"])
    pad_compare([b"1", b"2", b"3"], [5], None, [b"1", b"2", b"3", b"", b""])


def test_pad_end_exceptions():
    with pytest.raises(RuntimeError) as info:
        pad_compare([1, 2], [3], "-1", [])
    assert "Source and pad_value are not of the same type." in str(info.value)

    with pytest.raises(RuntimeError) as info:
        pad_compare([b"1", b"2", b"3", b"4", b"5"], [2], 1, [])
    assert "Source and pad_value are not of the same type." in str(info.value)

    with pytest.raises(TypeError) as info:
        pad_compare([3, 4, 5], ["2"], 1, [])
    assert "a value in the list is not an integer." in str(info.value)

    with pytest.raises(TypeError) as info:
        pad_compare([1, 2], 3, -1, [1, 2, -1])
    assert "Argument pad_end with value 3 is not of type (<class 'list'>,)" in str(info.value)


if __name__ == "__main__":
    test_pad_end_basics()
    test_pad_end_str()
    test_pad_end_exceptions()
