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

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
from mindspore.dataset.text import to_str, to_bytes


def test_basic():
    x = np.array([["ab", "cde", "121"], ["x", "km", "789"]], dtype='S')
    n = cde.Tensor(x)
    arr = n.as_array()
    np.testing.assert_array_equal(x, arr)


def compare(strings, dtype='S'):
    arr = np.array(strings, dtype=dtype)

    def gen():
        (yield arr,)

    data = ds.GeneratorDataset(gen, column_names=["col"])

    for d in data:
        np.testing.assert_array_equal(d[0], arr.astype('S'))


def test_generator():
    compare(["ab"])
    compare(["", ""])
    compare([""])
    compare(["ab", ""])
    compare(["ab", "cde", "121"])
    compare([["ab", "cde", "121"], ["x", "km", "789"]])
    compare([["ab", "", "121"], ["", "km", "789"]])
    compare(["ab"], dtype='U')
    compare(["", ""], dtype='U')
    compare([""], dtype='U')
    compare(["ab", ""], dtype='U')
    compare(["", ""], dtype='U')
    compare(["", "ab"], dtype='U')
    compare(["ab", "cde", "121"], dtype='U')
    compare([["ab", "cde", "121"], ["x", "km", "789"]], dtype='U')
    compare([["ab", "", "121"], ["", "km", "789"]], dtype='U')


line = np.array(["This is a text file.",
                 "Be happy every day.",
                 "Good luck to everyone."])

words = np.array([["This", "text", "file", "a"],
                  ["Be", "happy", "day", "b"],
                  ["女", "", "everyone", "c"]])

chinese = np.array(["今天天气太好了我们一起去外面玩吧",
                    "男默女泪",
                    "江州市长江大桥参加了长江大桥的通车仪式"])


def test_batching_strings():
    def gen():
        for row in chinese:
            yield (np.array(row),)

    data = ds.GeneratorDataset(gen, column_names=["col"])
    data = data.batch(2, drop_remainder=True)

    for d in data:
        np.testing.assert_array_equal(d[0], to_bytes(chinese[0:2]))


def test_map():
    def gen():
        yield (np.array(["ab cde 121"], dtype='S'),)

    data = ds.GeneratorDataset(gen, column_names=["col"])

    def split(b):
        s = to_str(b)
        splits = s.item().split()
        return np.array(splits)

    data = data.map(input_columns=["col"], operations=split)
    expected = np.array(["ab", "cde", "121"], dtype='S')
    for d in data:
        np.testing.assert_array_equal(d[0], expected)


def test_map2():
    def gen():
        yield (np.array(["ab cde 121"], dtype='S'),)

    data = ds.GeneratorDataset(gen, column_names=["col"])

    def upper(b):
        out = np.char.upper(b)
        return out

    data = data.map(input_columns=["col"], operations=upper)
    expected = np.array(["AB CDE 121"], dtype='S')
    for d in data:
        np.testing.assert_array_equal(d[0], expected)


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


# The following tests cases were copied from test_pad_batch but changed to strings instead


# this generator function yield two columns
# col1d: [0],[1], [2], [3]
# col2d: [[100],[200]], [[101],[201]], [102],[202]], [103],[203]]
def gen_2cols(num):
    for i in range(num):
        yield (np.array([str(i)]), np.array([[str(i + 100)], [str(i + 200)]]))


# this generator function yield one column of variable shapes
# col: [0], [0,1], [0,1,2], [0,1,2,3]
def gen_var_col(num):
    for i in range(num):
        yield (np.array([str(j) for j in range(i + 1)]),)


# this generator function yield two columns of variable shapes
# col1: [0], [0,1], [0,1,2], [0,1,2,3]
# col2: [100], [100,101], [100,101,102], [100,110,102,103]
def gen_var_cols(num):
    for i in range(num):
        yield (np.array([str(j) for j in range(i + 1)]), np.array([str(100 + j) for j in range(i + 1)]))


# this generator function yield two columns of variable shapes
# col1: [[0]], [[0,1]], [[0,1,2]], [[0,1,2,3]]
# col2: [[100]], [[100,101]], [[100,101,102]], [[100,110,102,103]]
def gen_var_cols_2d(num):
    for i in range(num):
        yield (np.array([[str(j) for j in range(i + 1)]]), np.array([[str(100 + j) for j in range(i + 1)]]))


def test_batch_padding_01():
    data1 = ds.GeneratorDataset((lambda: gen_2cols(2)), ["col1d", "col2d"])
    data1 = data1.batch(batch_size=2, drop_remainder=False, pad_info={"col2d": ([2, 2], b"-2"), "col1d": ([2], b"-1")})
    data1 = data1.repeat(2)
    for data in data1.create_dict_iterator():
        np.testing.assert_array_equal([[b"0", b"-1"], [b"1", b"-1"]], data["col1d"])
        np.testing.assert_array_equal([[[b"100", b"-2"], [b"200", b"-2"]], [[b"101", b"-2"], [b"201", b"-2"]]],
                                      data["col2d"])


def test_batch_padding_02():
    data1 = ds.GeneratorDataset((lambda: gen_2cols(2)), ["col1d", "col2d"])
    data1 = data1.batch(batch_size=2, drop_remainder=False, pad_info={"col2d": ([1, 2], "")})
    data1 = data1.repeat(2)
    for data in data1.create_dict_iterator():
        np.testing.assert_array_equal([[b"0"], [b"1"]], data["col1d"])
        np.testing.assert_array_equal([[[b"100", b""]], [[b"101", b""]]], data["col2d"])


def test_batch_padding_03():
    data1 = ds.GeneratorDataset((lambda: gen_var_col(4)), ["col"])
    data1 = data1.batch(batch_size=2, drop_remainder=False, pad_info={"col": (None, "PAD_VALUE")})  # pad automatically
    data1 = data1.repeat(2)
    res = dict()
    for ind, data in enumerate(data1.create_dict_iterator()):
        res[ind] = data["col"].copy()
    np.testing.assert_array_equal(res[0], [[b"0", b"PAD_VALUE"], [0, 1]])
    np.testing.assert_array_equal(res[1], [[b"0", b"1", b"2", b"PAD_VALUE"], [b"0", b"1", b"2", b"3"]])
    np.testing.assert_array_equal(res[2], [[b"0", b"PAD_VALUE"], [b"0", b"1"]])
    np.testing.assert_array_equal(res[3], [[b"0", b"1", b"2", b"PAD_VALUE"], [b"0", b"1", b"2", b"3"]])


def test_batch_padding_04():
    data1 = ds.GeneratorDataset((lambda: gen_var_cols(2)), ["col1", "col2"])
    data1 = data1.batch(batch_size=2, drop_remainder=False, pad_info={})  # pad automatically
    data1 = data1.repeat(2)
    for data in data1.create_dict_iterator():
        np.testing.assert_array_equal(data["col1"], [[b"0", b""], [b"0", b"1"]])
        np.testing.assert_array_equal(data["col2"], [[b"100", b""], [b"100", b"101"]])


def test_batch_padding_05():
    data1 = ds.GeneratorDataset((lambda: gen_var_cols_2d(3)), ["col1", "col2"])
    data1 = data1.batch(batch_size=3, drop_remainder=False,
                        pad_info={"col2": ([2, None], "-2"), "col1": (None, "-1")})  # pad automatically
    for data in data1.create_dict_iterator():
        np.testing.assert_array_equal(data["col1"],
                                      [[[b"0", b"-1", b"-1"]], [[b"0", b"1", b"-1"]], [[b"0", b"1", b"2"]]])
        np.testing.assert_array_equal(data["col2"],
                                      [[[b"100", b"-2", b"-2"], [b"-2", b"-2", b"-2"]],
                                       [[b"100", b"101", b"-2"], [b"-2", b"-2", b"-2"]],
                                       [[b"100", b"101", b"102"], [b"-2", b"-2", b"-2"]]])


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
    test_batch_padding_01()
    test_batch_padding_02()
    test_batch_padding_03()
    test_batch_padding_04()
    test_batch_padding_05()
