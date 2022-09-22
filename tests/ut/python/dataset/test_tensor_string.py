# Copyright 2019-2022 Huawei Technologies Co., Ltd
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

import mindspore
import mindspore._c_dataengine as cde
import mindspore.common.dtype as mstype
import mindspore.dataset as ds


def test_basic():
    """
    Feature: Tensor
    Description: Test basic Tensor op on NumPy dataset with strings
    Expectation: Output is equal to the expected output
    """
    byte_data = np.array([["ab", "cde", "121"], ["x", "km", "789"]], dtype=np.bytes_)
    byte_tensor = cde.Tensor(byte_data)
    byte_array = byte_tensor.as_array()
    np.testing.assert_array_equal(byte_data, byte_array)

    string_data = np.array([["ab", "cde", "121"], ["x", "km", "789"]], dtype=np.str_)
    string_tensor = cde.Tensor(string_data)
    string_array = string_tensor.as_array()
    np.testing.assert_array_equal(string_data, string_array)


def compare(strings, dtype="S"):
    arr = np.array(strings, dtype=dtype)

    def gen():
        (yield arr,)

    data = ds.GeneratorDataset(gen, column_names=["col"])

    for d in data.create_tuple_iterator(num_epochs=1, output_numpy=True):
        np.testing.assert_array_equal(d[0], arr)


def test_generator():
    """
    Feature: Tensor
    Description: Test string tensor with various valid inputs using GeneratorDataset
    Expectation: Output is equal to the expected output
    """
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
    """
    Feature: Tensor
    Description: Test applying Batch op to string tensor using GeneratorDataset
    Expectation: Output is equal to the expected output
    """

    def gen():
        for row in chinese:
            yield (np.array(row),)

    data = ds.GeneratorDataset(gen, column_names=["col"])
    data = data.batch(2, drop_remainder=True)

    for d in data.create_tuple_iterator(num_epochs=1, output_numpy=True):
        np.testing.assert_array_equal(d[0], chinese[0:2])


def test_map():
    """
    Feature: Tensor
    Description: Test applying Map op split to string tensor using GeneratorDataset
    Expectation: Output is equal to the expected output
    """

    def gen():
        yield (np.array(["ab cde 121"], dtype=np.str_),)

    data = ds.GeneratorDataset(gen, column_names=["col"])

    def split(s):
        splits = s.item().split()
        return np.array(splits)

    data = data.map(operations=split, input_columns=["col"])
    expected = np.array(["ab", "cde", "121"], dtype=np.str_)
    for d in data.create_tuple_iterator(num_epochs=1, output_numpy=True):
        np.testing.assert_array_equal(d[0], expected)


def test_map2():
    """
    Feature: Tensor
    Description: Test applying Map op upper to string tensor using GeneratorDataset
    Expectation: Output is equal to the expected output
    """

    def gen():
        yield (np.array(["ab cde 121"], dtype='S'),)

    data = ds.GeneratorDataset(gen, column_names=["col"])

    def upper(b):
        out = np.char.upper(b)
        return out

    data = data.map(operations=upper, input_columns=["col"])
    expected = np.array(["AB CDE 121"], dtype='S')
    for d in data.create_tuple_iterator(num_epochs=1, output_numpy=True):
        np.testing.assert_array_equal(d[0], expected)


def test_tfrecord1():
    """
    Feature: Tensor
    Description: Test string tensor using TFRecordDataset with created schema using "string" type
    Expectation: Output is equal to the expected output
    """
    s = ds.Schema()
    s.add_column("line", "string", [])
    s.add_column("words", "string", [-1])
    s.add_column("chinese", "string", [])

    data = ds.TFRecordDataset("../data/dataset/testTextTFRecord/text.tfrecord", shuffle=False, schema=s)

    for i, d in enumerate(data.create_dict_iterator(num_epochs=1, output_numpy=True)):
        assert d["line"].shape == line[i].shape
        assert d["words"].shape == words[i].shape
        assert d["chinese"].shape == chinese[i].shape
        np.testing.assert_array_equal(line[i], d["line"])
        np.testing.assert_array_equal(words[i], d["words"])
        np.testing.assert_array_equal(chinese[i], d["chinese"])


def test_tfrecord2():
    """
    Feature: Tensor
    Description: Test string tensor using TFRecordDataset with schema from a file
    Expectation: Output is equal to the expected output
    """
    data = ds.TFRecordDataset("../data/dataset/testTextTFRecord/text.tfrecord", shuffle=False,
                              schema='../data/dataset/testTextTFRecord/datasetSchema.json')
    for i, d in enumerate(data.create_dict_iterator(num_epochs=1, output_numpy=True)):
        assert d["line"].shape == line[i].shape
        assert d["words"].shape == words[i].shape
        assert d["chinese"].shape == chinese[i].shape
        np.testing.assert_array_equal(line[i], d["line"])
        np.testing.assert_array_equal(words[i], d["words"])
        np.testing.assert_array_equal(chinese[i], d["chinese"])


def test_tfrecord3():
    """
    Feature: Tensor
    Description: Test string tensor using TFRecordDataset with created schema using mstype.string type
    Expectation: Output is equal to the expected output
    """
    s = ds.Schema()
    s.add_column("line", mstype.string, [])
    s.add_column("words", mstype.string, [-1, 2])
    s.add_column("chinese", mstype.string, [])

    data = ds.TFRecordDataset("../data/dataset/testTextTFRecord/text.tfrecord", shuffle=False, schema=s)

    for i, d in enumerate(data.create_dict_iterator(num_epochs=1, output_numpy=True)):
        assert d["line"].shape == line[i].shape
        assert d["words"].shape == words[i].reshape([2, 2]).shape
        assert d["chinese"].shape == chinese[i].shape
        np.testing.assert_array_equal(line[i], d["line"])
        np.testing.assert_array_equal(words[i].reshape([2, 2]), d["words"])
        np.testing.assert_array_equal(chinese[i], d["chinese"])


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
    """
    Feature: Tensor
    Description: Test string tensor using MindDataset
    Expectation: Output is equal to the expected output
    """
    data = ds.MindDataset("../data/dataset/testTextMindRecord/test.mindrecord", shuffle=False)

    for i, d in enumerate(data.create_dict_iterator(num_epochs=1, output_numpy=True)):
        assert d["english"].shape == line[i].shape
        assert d["chinese"].shape == chinese[i].shape
        np.testing.assert_array_equal(line[i], d["english"])
        np.testing.assert_array_equal(chinese[i], d["chinese"])


# The following tests cases were copied from test_pad_batch but changed to strings instead


# this generator function yield two columns
# col1d: [0],[1], [2], [3]
# col2d: [[100],[200]], [[101],[201]], [102],[202]], [103],[203]]
def gen_2cols(num):
    for i in range(num):
        yield np.array([str(i)], dtype=np.str_), np.array([[str(i + 100)], [str(i + 200)]], dtype=np.bytes_)


# this generator function yield one column of variable shapes
# col: [0], [0,1], [0,1,2], [0,1,2,3]
def gen_var_col(num):
    for i in range(num):
        yield np.array([str(j) for j in range(i + 1)])


# this generator function yield two columns of variable shapes
# col1: [0], [0,1], [0,1,2], [0,1,2,3]
# col2: [100], [100,101], [100,101,102], [100,110,102,103]
def gen_var_cols(num):
    for i in range(num):
        yield np.array([str(j) for j in range(i + 1)]), np.array([str(100 + j) for j in range(i + 1)])


# this generator function yield two columns of variable shapes
# col1: [[0]], [[0,1]], [[0,1,2]], [[0,1,2,3]]
# col2: [[100]], [[100,101]], [[100,101,102]], [[100,110,102,103]]
def gen_var_cols_2d(num):
    for i in range(num):
        yield (np.array([[str(j) for j in range(i + 1)]], dtype=np.str_),
               np.array([[str(100 + j) for j in range(i + 1)]], dtype=np.bytes_))


def test_batch_padding_01():
    """
    Feature: Batch Padding
    Description: Test batch padding where input_shape=[x] and output_shape=[y] in which y > x
    Expectation: Output is equal to the expected output
    """
    data1 = ds.GeneratorDataset((lambda: gen_2cols(2)), ["col1d", "col2d"])
    data1 = data1.padded_batch(batch_size=2, drop_remainder=False,
                               pad_info={"col2d": ([2, 2], b"-2"), "col1d": ([2], "-1")})
    data1 = data1.repeat(2)
    for data in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        np.testing.assert_array_equal([["0", "-1"], ["1", "-1"]], data["col1d"])
        np.testing.assert_array_equal([[[b"100", b"-2"], [b"200", b"-2"]], [[b"101", b"-2"], [b"201", b"-2"]]],
                                      data["col2d"])


def test_batch_padding_02():
    """
    Feature: Batch Padding
    Description: Test batch padding where padding in one dimension and truncate in the other, in which
        input_shape=[x1,x2] and output_shape=[y1,y2] and y1 > x1 and y2 < x2
    Expectation: Output is equal to the expected output
    """
    data1 = ds.GeneratorDataset((lambda: gen_2cols(2)), ["col1d", "col2d"])
    data1 = data1.padded_batch(batch_size=2, drop_remainder=False, pad_info={"col2d": ([1, 2], b"")})
    data1 = data1.repeat(2)
    for data in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        np.testing.assert_array_equal([["0"], ["1"]], data["col1d"])
        np.testing.assert_array_equal([[[b"100", b""]], [[b"101", b""]]], data["col2d"])


def test_batch_padding_03():
    """
    Feature: Batch Padding
    Description: Test batch padding using automatic padding for a specific column
    Expectation: Output is equal to the expected output
    """
    data1 = ds.GeneratorDataset((lambda: gen_var_col(4)), ["col"])
    data1 = data1.padded_batch(batch_size=2, drop_remainder=False, pad_info={"col": (None, "PAD_VALUE")})
    data1 = data1.repeat(2)
    res = []
    for data in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        res.append(data["col"].copy())
    np.testing.assert_array_equal(res[0], [["0", "PAD_VALUE"], [0, 1]])
    np.testing.assert_array_equal(res[1], [["0", "1", "2", "PAD_VALUE"], ["0", "1", "2", "3"]])
    np.testing.assert_array_equal(res[2], [["0", "PAD_VALUE"], ["0", "1"]])
    np.testing.assert_array_equal(res[3], [["0", "1", "2", "PAD_VALUE"], ["0", "1", "2", "3"]])


def test_batch_padding_04():
    """
    Feature: Batch Padding
    Description: Test batch padding using default setting for all columns
    Expectation: Output is equal to the expected output
    """
    data1 = ds.GeneratorDataset((lambda: gen_var_cols(2)), ["col1", "col2"])
    data1 = data1.padded_batch(batch_size=2, drop_remainder=False, pad_info={})  # pad automatically
    data1 = data1.repeat(2)
    for data in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        np.testing.assert_array_equal(data["col1"], [["0", ""], ["0", "1"]])
        np.testing.assert_array_equal(data["col2"], [["100", ""], ["100", "101"]])


def test_batch_padding_05():
    """
    Feature: Batch Padding
    Description: Test batch padding where None is in different places
    Expectation: Output is equal to the expected output
    """
    data1 = ds.GeneratorDataset((lambda: gen_var_cols_2d(3)), ["col1", "col2"])
    data1 = data1.padded_batch(batch_size=3, drop_remainder=False,
                               pad_info={"col2": ([2, None], b"-2"), "col1": (None, "-1")})
    for data in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        np.testing.assert_array_equal(data["col1"],
                                      [[["0", "-1", "-1"]], [["0", "1", "-1"]], [["0", "1", "2"]]])
        np.testing.assert_array_equal(data["col2"],
                                      [[[b"100", b"-2", b"-2"], [b"-2", b"-2", b"-2"]],
                                       [[b"100", b"101", b"-2"], [b"-2", b"-2", b"-2"]],
                                       [[b"100", b"101", b"102"], [b"-2", b"-2", b"-2"]]])


def test_process_string_pipeline():
    """
    Feature: String and Bytes Tensor
    Description: Test processing string and bytes data
    Expectation: The output is as expected
    """
    def generate_and_process_string(dtype):
        data = np.array([["apple"], ["orange"], ["banana"], ["1"], ["2"], ["3"], ["a"], ["b"], ["c"]], dtype=dtype)
        dataset = ds.NumpySlicesDataset(data, column_names=["text"])
        assert dataset.output_types()[0].type == dtype
        dataset = dataset.map(lambda e: (e, e), input_columns=["text"], output_columns=["text1", "text2"],
                              column_order=["text1", "text2"])
        for i, item in enumerate(dataset.create_dict_iterator(num_epochs=1, output_numpy=True)):
            item["text1"] = data[i]
            item["text2"] = data[i]
        for i, item in enumerate(dataset.create_tuple_iterator(num_epochs=1)):
            item[0] = mindspore.Tensor(data[i])
            item[1] = mindspore.Tensor(data[i])

    generate_and_process_string(np.bytes_)
    generate_and_process_string(np.str_)


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
    test_process_string_pipeline()
