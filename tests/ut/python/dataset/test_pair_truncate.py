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
Testing Mask op in DE
"""
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.text as text


def compare(in1, in2, length, out1, out2):
    data = ds.NumpySlicesDataset({"s1": [in1], "s2": [in2]})
    data = data.map(operations=text.TruncateSequencePair(length), input_columns=["s1", "s2"])
    data = data.map(input_columns=["s1", "s2"], operations=text.TruncateSequencePair(length))
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        np.testing.assert_array_equal(out1, d["s1"])
        np.testing.assert_array_equal(out2, d["s2"])


def test_callable():
    op = text.TruncateSequencePair(3)
    data = [["1", "2", "3"], ["4", "5"]]
    result_text = op(*data)
    column1, column2 = op(["1", "2", "3"], ["4", "5"])
    assert np.array_equal(result_text[0], ['1', '2'])
    assert np.array_equal(result_text[1], ['4'])
    assert np.array_equal(column1, ['1', '2'])
    assert np.array_equal(column2, ['4'])


def test_basics():
    compare(in1=[1, 2, 3], in2=[4, 5], length=4, out1=[1, 2], out2=[4, 5])
    compare(in1=[1, 2], in2=[4, 5], length=4, out1=[1, 2], out2=[4, 5])
    compare(in1=[1], in2=[4], length=4, out1=[1], out2=[4])
    compare(in1=[1, 2, 3, 4], in2=[5], length=4, out1=[1, 2, 3], out2=[5])
    compare(in1=[1, 2, 3, 4], in2=[5, 6, 7, 8], length=4, out1=[1, 2], out2=[5, 6])


def test_basics_odd():
    compare(in1=[1, 2, 3], in2=[4, 5], length=3, out1=[1, 2], out2=[4])
    compare(in1=[1, 2], in2=[4, 5], length=3, out1=[1, 2], out2=[4])
    compare(in1=[1], in2=[4], length=5, out1=[1], out2=[4])
    compare(in1=[1, 2, 3, 4], in2=[5], length=3, out1=[1, 2], out2=[5])
    compare(in1=[1, 2, 3, 4], in2=[5, 6, 7, 8], length=3, out1=[1, 2], out2=[5])


def test_basics_str():
    compare(in1=[b"1", b"2", b"3"], in2=[4, 5], length=4, out1=[b"1", b"2"], out2=[4, 5])
    compare(in1=[b"1", b"2"], in2=[b"4", b"5"], length=4, out1=[b"1", b"2"], out2=[b"4", b"5"])
    compare(in1=[b"1"], in2=[4], length=4, out1=[b"1"], out2=[4])
    compare(in1=[b"1", b"2", b"3", b"4"], in2=[b"5"], length=4, out1=[b"1", b"2", b"3"], out2=[b"5"])
    compare(in1=[b"1", b"2", b"3", b"4"], in2=[5, 6, 7, 8], length=4, out1=[b"1", b"2"], out2=[5, 6])


def test_exceptions():
    compare(in1=[1, 2, 3, 4], in2=[5, 6, 7, 8], length=1, out1=[1], out2=[])


if __name__ == "__main__":
    test_callable()
    test_basics()
    test_basics_odd()
    test_basics_str()
    test_exceptions()
