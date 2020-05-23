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
import numpy as np
import pytest

from mindspore.dataset.text import to_str, to_bytes

import mindspore.dataset as ds
import mindspore.common.dtype as mstype


def test_basic():
    x = np.array([["ab", "cde", "121"], ["x", "km", "789"]], dtype='S')
    n = cde.Tensor(x)
    arr = n.as_array()
    np.testing.assert_array_equal(x, arr)


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
    assert "[Batch ERROR] Batch does not support" in str(info.value)


def test_map():
    def gen():
        yield np.array(["ab cde 121"], dtype='S'),

    data = ds.GeneratorDataset(gen, column_names=["col"])

    def split(b):
        s = to_str(b)
        splits = s.item().split()
        return np.array(splits, dtype='S')

    data = data.map(input_columns=["col"], operations=split)
    expected = np.array(["ab", "cde", "121"], dtype='S')
    for d in data:
        np.testing.assert_array_equal(d[0], expected)


def test_map2():
    def gen():
        yield np.array(["ab cde 121"], dtype='S'),

    data = ds.GeneratorDataset(gen, column_names=["col"])

    def upper(b):
        out = np.char.upper(b)
        return out

    data = data.map(input_columns=["col"], operations=upper)
    expected = np.array(["AB CDE 121"], dtype='S')
    for d in data:
        np.testing.assert_array_equal(d[0], expected)


line = np.array(["This is a text file.",
                 "Be happy every day.",
                 "Good luck to everyone."])

words = np.array([["This", "text", "file", "a"],
                  ["Be", "happy", "day", "b"],
                  ["女", "", "everyone", "c"]])

chinese = np.array(["今天天气太好了我们一起去外面玩吧",
                    "男默女泪",
                    "江州市长江大桥参加了长江大桥的通车仪式"])


def test_tfrecord1():
    s = ds.Schema()
    s.add_column("line", "string", [])
    s.add_column("words", "string", [-1])
    s.add_column("chinese", "string", [])

    data = ds.TFRecordDataset("../data/dataset/testTextTFRecord/text.tfrecord", shuffle=False, schema=s)

    for i, d in enumerate(data.create_dict_iterator()):
        assert d["line"].shape == line[i].shape
        assert d["words"].shape == words[i].shape
        assert d["chinese"].shape == chinese[i].shape
        np.testing.assert_array_equal(line[i], to_str(d["line"]))
        np.testing.assert_array_equal(words[i], to_str(d["words"]))
        np.testing.assert_array_equal(chinese[i], to_str(d["chinese"]))


def test_tfrecord2():
    data = ds.TFRecordDataset("../data/dataset/testTextTFRecord/text.tfrecord", shuffle=False,
                              schema='../data/dataset/testTextTFRecord/datasetSchema.json')
    for i, d in enumerate(data.create_dict_iterator()):
        assert d["line"].shape == line[i].shape
        assert d["words"].shape == words[i].shape
        assert d["chinese"].shape == chinese[i].shape
        np.testing.assert_array_equal(line[i], to_str(d["line"]))
        np.testing.assert_array_equal(words[i], to_str(d["words"]))
        np.testing.assert_array_equal(chinese[i], to_str(d["chinese"]))


def test_tfrecord3():
    s = ds.Schema()
    s.add_column("line", mstype.string, [])
    s.add_column("words", mstype.string, [-1, 2])
    s.add_column("chinese", mstype.string, [])

    data = ds.TFRecordDataset("../data/dataset/testTextTFRecord/text.tfrecord", shuffle=False, schema=s)

    for i, d in enumerate(data.create_dict_iterator()):
        assert d["line"].shape == line[i].shape
        assert d["words"].shape == words[i].reshape([2, 2]).shape
        assert d["chinese"].shape == chinese[i].shape
        np.testing.assert_array_equal(line[i], to_str(d["line"]))
        np.testing.assert_array_equal(words[i].reshape([2, 2]), to_str(d["words"]))
        np.testing.assert_array_equal(chinese[i], to_str(d["chinese"]))


def create_text_mindrecord():
    # methood to create mindrecord with string data, used to generate testTextMindRecord/test.mindrecord
    from mindspore.mindrecord import FileWriter

    mindrecord_file_name = "test.mindrecord"
    data = [{"english": "This is a text file.",
             "chinese": "今天天气太好了我们一起去外面玩吧"},
            {"english": "Be happy every day.",
             "chinese": "男默女泪"},
            {"english": "Good luck to everyone.",
             "chinese": "江州市长江大桥参加了长江大桥的通车仪式"},
            ]
    writer = FileWriter(mindrecord_file_name)
    schema = {"english": {"type": "string"},
              "chinese": {"type": "string"},
              }
    writer.add_schema(schema)
    writer.write_raw_data(data)
    writer.commit()


def test_mindrecord():
    data = ds.MindDataset("../data/dataset/testTextMindRecord/test.mindrecord", shuffle=False)

    for i, d in enumerate(data.create_dict_iterator()):
        assert d["english"].shape == line[i].shape
        assert d["chinese"].shape == chinese[i].shape
        np.testing.assert_array_equal(line[i], to_str(d["english"]))
        np.testing.assert_array_equal(chinese[i], to_str(d["chinese"]))


if __name__ == '__main__':
    test_generator()
    test_basic()
    test_batching_strings()
    test_map()
    test_map2()
    test_tfrecord1()
    test_tfrecord2()
    test_tfrecord3()
    test_mindrecord()
