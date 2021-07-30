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
This is the test module for saveOp.
"""
import os
from string import punctuation
import numpy as np
import pytest
import mindspore.dataset as ds
from mindspore import log as logger
from mindspore.mindrecord import FileWriter

TEMP_FILE = "../data/mindrecord/testMindDataSet/temp.mindrecord"
AUTO_FILE = "../data/mindrecord/testMindDataSet/auto.mindrecord"
TFRECORD_FILES = "../data/mindrecord/testTFRecordData/dummy.tfrecord"
FILES_NUM = 1
num_readers = 1


@pytest.fixture(name="add_remove_file")
def fixture_remove():
    """add/remove cv file"""
    if os.path.exists("{}".format(TEMP_FILE)):
        os.remove("{}".format(TEMP_FILE))
    if os.path.exists("{}.db".format(TEMP_FILE)):
        os.remove("{}.db".format(TEMP_FILE))

    if os.path.exists("{}".format(AUTO_FILE)):
        os.remove("{}".format(AUTO_FILE))
    if os.path.exists("{}.db".format(AUTO_FILE)):
        os.remove("{}.db".format(AUTO_FILE))
    yield "yield_cv_data"
    if os.path.exists("{}".format(TEMP_FILE)):
        os.remove("{}".format(TEMP_FILE))
    if os.path.exists("{}.db".format(TEMP_FILE)):
        os.remove("{}.db".format(TEMP_FILE))

    if os.path.exists("{}".format(AUTO_FILE)):
        os.remove("{}".format(AUTO_FILE))
    if os.path.exists("{}.db".format(AUTO_FILE)):
        os.remove("{}.db".format(AUTO_FILE))


def test_case_00(add_remove_file):  # only bin data
    data = [{"image1": bytes("image1 bytes abc", encoding='UTF-8'),
             "image2": bytes("image1 bytes def", encoding='UTF-8'),
             "image3": bytes("image1 bytes ghi", encoding='UTF-8'),
             "image4": bytes("image1 bytes jkl", encoding='UTF-8'),
             "image5": bytes("image1 bytes mno", encoding='UTF-8')},
            {"image1": bytes("image2 bytes abc", encoding='UTF-8'),
             "image2": bytes("image2 bytes def", encoding='UTF-8'),
             "image3": bytes("image2 bytes ghi", encoding='UTF-8'),
             "image4": bytes("image2 bytes jkl", encoding='UTF-8'),
             "image5": bytes("image2 bytes mno", encoding='UTF-8')},
            {"image1": bytes("image3 bytes abc", encoding='UTF-8'),
             "image2": bytes("image3 bytes def", encoding='UTF-8'),
             "image3": bytes("image3 bytes ghi", encoding='UTF-8'),
             "image4": bytes("image3 bytes jkl", encoding='UTF-8'),
             "image5": bytes("image3 bytes mno", encoding='UTF-8')},
            {"image1": bytes("image5 bytes abc", encoding='UTF-8'),
             "image2": bytes("image5 bytes def", encoding='UTF-8'),
             "image3": bytes("image5 bytes ghi", encoding='UTF-8'),
             "image4": bytes("image5 bytes jkl", encoding='UTF-8'),
             "image5": bytes("image5 bytes mno", encoding='UTF-8')},
            {"image1": bytes("image6 bytes abc", encoding='UTF-8'),
             "image2": bytes("image6 bytes def", encoding='UTF-8'),
             "image3": bytes("image6 bytes ghi", encoding='UTF-8'),
             "image4": bytes("image6 bytes jkl", encoding='UTF-8'),
             "image5": bytes("image6 bytes mno", encoding='UTF-8')}]
    schema = {
        "image1": {"type": "bytes"},
        "image2": {"type": "bytes"},
        "image3": {"type": "bytes"},
        "image4": {"type": "bytes"},
        "image5": {"type": "bytes"}}
    writer = FileWriter(TEMP_FILE, FILES_NUM)
    writer.add_schema(schema, "schema")
    writer.write_raw_data(data)
    writer.commit()

    d1 = ds.MindDataset(TEMP_FILE, None, num_readers, shuffle=False)
    d1.save(AUTO_FILE, FILES_NUM)
    data_value_to_list = []

    for item in data:
        new_data = {}
        new_data['image1'] = np.asarray(list(item["image1"]), dtype=np.uint8)
        new_data['image2'] = np.asarray(list(item["image2"]), dtype=np.uint8)
        new_data['image3'] = np.asarray(list(item["image3"]), dtype=np.uint8)
        new_data['image4'] = np.asarray(list(item["image4"]), dtype=np.uint8)
        new_data['image5'] = np.asarray(list(item["image5"]), dtype=np.uint8)
        data_value_to_list.append(new_data)

    d2 = ds.MindDataset(dataset_file=AUTO_FILE,
                        num_parallel_workers=num_readers,
                        shuffle=False)
    assert d2.get_dataset_size() == 5
    num_iter = 0
    for item in d2.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 5
        for field in item:
            if isinstance(item[field], np.ndarray):
                assert (item[field] ==
                        data_value_to_list[num_iter][field]).all()
            else:
                assert item[field] == data_value_to_list[num_iter][field]
        num_iter += 1
    assert num_iter == 5


def test_case_01(add_remove_file):  # only raw data
    data = [{"file_name": "001.jpg", "label": 43},
            {"file_name": "002.jpg", "label": 91},
            {"file_name": "003.jpg", "label": 61},
            {"file_name": "004.jpg", "label": 29},
            {"file_name": "005.jpg", "label": 78},
            {"file_name": "006.jpg", "label": 37}]
    schema = {"file_name": {"type": "string"},
              "label": {"type": "int32"}
              }

    writer = FileWriter(TEMP_FILE, FILES_NUM)
    writer.add_schema(schema, "schema")
    writer.write_raw_data(data)
    writer.commit()

    d1 = ds.MindDataset(TEMP_FILE, None, num_readers, shuffle=False)
    d1.save(AUTO_FILE, FILES_NUM)

    data_value_to_list = []
    for item in data:
        new_data = {}
        new_data['file_name'] = np.asarray(item["file_name"], dtype='S')
        new_data['label'] = np.asarray(list([item["label"]]), dtype=np.int32)
        data_value_to_list.append(new_data)

    d2 = ds.MindDataset(dataset_file=AUTO_FILE,
                        num_parallel_workers=num_readers,
                        shuffle=False)
    assert d2.get_dataset_size() == 6
    num_iter = 0
    for item in d2.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(item)
        assert len(item) == 2
        for field in item:
            if isinstance(item[field], np.ndarray):
                assert (item[field] ==
                        data_value_to_list[num_iter][field]).all()
            else:
                assert item[field] == data_value_to_list[num_iter][field]
        num_iter += 1
    assert num_iter == 6


def test_case_02(add_remove_file):  # muti-bytes
    data = [{"file_name": "001.jpg", "label": 43,
             "float32_array": np.array([1.2, 2.78, 3.1234, 4.9871, 5.12341], dtype=np.float32),
             "float64_array": np.array([48.1234556789, 49.3251241431, 50.13514312414, 51.8971298471,
                                        123414314.2141243, 87.1212122], dtype=np.float64),
             "float32": 3456.12345,
             "float64": 1987654321.123456785,
             "source_sos_ids": np.array([1, 2, 3, 4, 5], dtype=np.int32),
             "source_sos_mask": np.array([6, 7, 8, 9, 10, 11, 12], dtype=np.int64),
             "image1": bytes("image1 bytes abc", encoding='UTF-8'),
             "image2": bytes("image1 bytes def", encoding='UTF-8'),
             "image3": bytes("image1 bytes ghi", encoding='UTF-8'),
             "image4": bytes("image1 bytes jkl", encoding='UTF-8'),
             "image5": bytes("image1 bytes mno", encoding='UTF-8')},
            {"file_name": "002.jpg", "label": 91,
             "float32_array": np.array([1.2, 2.78, 4.1234, 4.9871, 5.12341], dtype=np.float32),
             "float64_array": np.array([48.1234556789, 49.3251241431, 60.13514312414, 51.8971298471,
                                        123414314.2141243, 87.1212122], dtype=np.float64),
             "float32": 3456.12445,
             "float64": 1987654321.123456786,
             "source_sos_ids": np.array([11, 2, 3, 4, 5], dtype=np.int32),
             "source_sos_mask": np.array([16, 7, 8, 9, 10, 11, 12], dtype=np.int64),
             "image1": bytes("image2 bytes abc", encoding='UTF-8'),
             "image2": bytes("image2 bytes def", encoding='UTF-8'),
             "image3": bytes("image2 bytes ghi", encoding='UTF-8'),
             "image4": bytes("image2 bytes jkl", encoding='UTF-8'),
             "image5": bytes("image2 bytes mno", encoding='UTF-8')},
            {"file_name": "003.jpg", "label": 61,
             "float32_array": np.array([1.2, 2.78, 5.1234, 4.9871, 5.12341], dtype=np.float32),
             "float64_array": np.array([48.1234556789, 49.3251241431, 70.13514312414, 51.8971298471,
                                        123414314.2141243, 87.1212122], dtype=np.float64),
             "float32": 3456.12545,
             "float64": 1987654321.123456787,
             "source_sos_ids": np.array([21, 2, 3, 4, 5], dtype=np.int32),
             "source_sos_mask": np.array([26, 7, 8, 9, 10, 11, 12], dtype=np.int64),
             "image1": bytes("image3 bytes abc", encoding='UTF-8'),
             "image2": bytes("image3 bytes def", encoding='UTF-8'),
             "image3": bytes("image3 bytes ghi", encoding='UTF-8'),
             "image4": bytes("image3 bytes jkl", encoding='UTF-8'),
             "image5": bytes("image3 bytes mno", encoding='UTF-8')},
            {"file_name": "004.jpg", "label": 29,
             "float32_array": np.array([1.2, 2.78, 6.1234, 4.9871, 5.12341], dtype=np.float32),
             "float64_array": np.array([48.1234556789, 49.3251241431, 80.13514312414, 51.8971298471,
                                        123414314.2141243, 87.1212122], dtype=np.float64),
             "float32": 3456.12645,
             "float64": 1987654321.123456788,
             "source_sos_ids": np.array([31, 2, 3, 4, 5], dtype=np.int32),
             "source_sos_mask": np.array([36, 7, 8, 9, 10, 11, 12], dtype=np.int64),
             "image1": bytes("image4 bytes abc", encoding='UTF-8'),
             "image2": bytes("image4 bytes def", encoding='UTF-8'),
             "image3": bytes("image4 bytes ghi", encoding='UTF-8'),
             "image4": bytes("image4 bytes jkl", encoding='UTF-8'),
             "image5": bytes("image4 bytes mno", encoding='UTF-8')},
            {"file_name": "005.jpg", "label": 78,
             "float32_array": np.array([1.2, 2.78, 7.1234, 4.9871, 5.12341], dtype=np.float32),
             "float64_array": np.array([48.1234556789, 49.3251241431, 90.13514312414, 51.8971298471,
                                        123414314.2141243, 87.1212122], dtype=np.float64),
             "float32": 3456.12745,
             "float64": 1987654321.123456789,
             "source_sos_ids": np.array([41, 2, 3, 4, 5], dtype=np.int32),
             "source_sos_mask": np.array([46, 7, 8, 9, 10, 11, 12], dtype=np.int64),
             "image1": bytes("image5 bytes abc", encoding='UTF-8'),
             "image2": bytes("image5 bytes def", encoding='UTF-8'),
             "image3": bytes("image5 bytes ghi", encoding='UTF-8'),
             "image4": bytes("image5 bytes jkl", encoding='UTF-8'),
             "image5": bytes("image5 bytes mno", encoding='UTF-8')},
            {"file_name": "006.jpg", "label": 37,
             "float32_array": np.array([1.2, 2.78, 7.1234, 4.9871, 5.12341], dtype=np.float32),
             "float64_array": np.array([48.1234556789, 49.3251241431, 90.13514312414, 51.8971298471,
                                        123414314.2141243, 87.1212122], dtype=np.float64),
             "float32": 3456.12745,
             "float64": 1987654321.123456789,
             "source_sos_ids": np.array([51, 2, 3, 4, 5], dtype=np.int32),
             "source_sos_mask": np.array([56, 7, 8, 9, 10, 11, 12], dtype=np.int64),
             "image1": bytes("image6 bytes abc", encoding='UTF-8'),
             "image2": bytes("image6 bytes def", encoding='UTF-8'),
             "image3": bytes("image6 bytes ghi", encoding='UTF-8'),
             "image4": bytes("image6 bytes jkl", encoding='UTF-8'),
             "image5": bytes("image6 bytes mno", encoding='UTF-8')}
            ]
    schema = {"file_name": {"type": "string"},
              "float32_array": {"type": "float32", "shape": [-1]},
              "float64_array": {"type": "float64", "shape": [-1]},
              "float32": {"type": "float32"},
              "float64": {"type": "float64"},
              "source_sos_ids": {"type": "int32", "shape": [-1]},
              "source_sos_mask": {"type": "int64", "shape": [-1]},
              "image1": {"type": "bytes"},
              "image2": {"type": "bytes"},
              "image3": {"type": "bytes"},
              "label": {"type": "int32"},
              "image4": {"type": "bytes"},
              "image5": {"type": "bytes"}}
    writer = FileWriter(TEMP_FILE, FILES_NUM)
    writer.add_schema(schema, "schema")
    writer.write_raw_data(data)
    writer.commit()

    d1 = ds.MindDataset(TEMP_FILE, None, num_readers, shuffle=False)
    d1.save(AUTO_FILE, FILES_NUM)
    data_value_to_list = []

    for item in data:
        new_data = {}
        new_data['file_name'] = np.asarray(item["file_name"], dtype='S')
        new_data['float32_array'] = item["float32_array"]
        new_data['float64_array'] = item["float64_array"]
        new_data['float32'] = item["float32"]
        new_data['float64'] = item["float64"]
        new_data['source_sos_ids'] = item["source_sos_ids"]
        new_data['source_sos_mask'] = item["source_sos_mask"]
        new_data['label'] = np.asarray(list([item["label"]]), dtype=np.int32)
        new_data['image1'] = np.asarray(list(item["image1"]), dtype=np.uint8)
        new_data['image2'] = np.asarray(list(item["image2"]), dtype=np.uint8)
        new_data['image3'] = np.asarray(list(item["image3"]), dtype=np.uint8)
        new_data['image4'] = np.asarray(list(item["image4"]), dtype=np.uint8)
        new_data['image5'] = np.asarray(list(item["image5"]), dtype=np.uint8)
        data_value_to_list.append(new_data)

    d2 = ds.MindDataset(dataset_file=AUTO_FILE,
                        num_parallel_workers=num_readers,
                        shuffle=False)
    assert d2.get_dataset_size() == 6
    num_iter = 0
    for item in d2.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 13
        for field in item:
            if isinstance(item[field], np.ndarray):
                if item[field].dtype == np.float32:
                    assert (item[field] ==
                            np.array(data_value_to_list[num_iter][field], np.float32)).all()
                else:
                    assert (item[field] ==
                            data_value_to_list[num_iter][field]).all()
            else:
                assert item[field] == data_value_to_list[num_iter][field]
        num_iter += 1
    assert num_iter == 6


def generator_1d():
    for i in range(10):
        yield (np.array([i]),)


def test_case_03(add_remove_file):

    # apply dataset operations
    d1 = ds.GeneratorDataset(generator_1d, ["data"], shuffle=False)

    d1.save(AUTO_FILE)

    d2 = ds.MindDataset(dataset_file=AUTO_FILE,
                        num_parallel_workers=num_readers,
                        shuffle=False)

    i = 0
    # each data is a dictionary
    for item in d2.create_dict_iterator(num_epochs=1, output_numpy=True):
        golden = np.array([i])
        np.testing.assert_array_equal(item["data"], golden)
        i = i + 1


def generator_with_type(t):
    for i in range(64):
        yield (np.array([i], dtype=t),)


def type_tester(t):
    logger.info("Test with Type {}".format(t.__name__))

    # apply dataset operations
    data1 = ds.GeneratorDataset((lambda: generator_with_type(t)), ["data"], shuffle=False)

    data1 = data1.batch(4)

    data1 = data1.repeat(3)

    data1.save(AUTO_FILE)

    d2 = ds.MindDataset(dataset_file=AUTO_FILE,
                        num_parallel_workers=num_readers,
                        shuffle=False)

    i = 0
    num_repeat = 0
    # each data is a dictionary
    for item in d2.create_dict_iterator(num_epochs=1, output_numpy=True):
        golden = np.array([[i], [i + 1], [i + 2], [i + 3]], dtype=t)
        logger.info(item)
        np.testing.assert_array_equal(item["data"], golden)
        i = i + 4
        if i == 64:
            i = 0
            num_repeat += 1
    assert num_repeat == 3
    if os.path.exists("{}".format(AUTO_FILE)):
        os.remove("{}".format(AUTO_FILE))
    if os.path.exists("{}.db".format(AUTO_FILE)):
        os.remove("{}.db".format(AUTO_FILE))


def test_case_04():
    # uint8 will drop shape as mindrecord store uint8 as bytes
    types = [np.int8, np.int16, np.int32, np.int64,
             np.uint16, np.uint32, np.float32, np.float64]

    for t in types:
        type_tester(t)


def test_case_05(add_remove_file):

    d1 = ds.GeneratorDataset(generator_1d, ["data"], shuffle=False)

    with pytest.raises(Exception, match="num_files should between 0 and 1000."):
        d1.save(AUTO_FILE, 0)


def test_case_06(add_remove_file):

    d1 = ds.GeneratorDataset(generator_1d, ["data"], shuffle=False)

    with pytest.raises(Exception, match="tfrecord dataset format is not supported."):
        d1.save(AUTO_FILE, 1, "tfrecord")


def cast_name(key):
    """
    Cast schema names which containing special characters to valid names.
    """
    special_symbols = set('{}{}'.format(punctuation, ' '))
    special_symbols.remove('_')
    new_key = ['_' if x in special_symbols else x for x in key]
    casted_key = ''.join(new_key)
    return casted_key


def test_case_07():
    if os.path.exists("{}".format(AUTO_FILE)):
        os.remove("{}".format(AUTO_FILE))
    if os.path.exists("{}.db".format(AUTO_FILE)):
        os.remove("{}.db".format(AUTO_FILE))
    d1 = ds.TFRecordDataset(TFRECORD_FILES, shuffle=False)
    tf_data = []
    for x in d1.create_dict_iterator(num_epochs=1, output_numpy=True):
        tf_data.append(x)
    d1.save(AUTO_FILE, FILES_NUM)
    d2 = ds.MindDataset(dataset_file=AUTO_FILE,
                        num_parallel_workers=num_readers,
                        shuffle=False)
    mr_data = []
    for x in d2.create_dict_iterator(num_epochs=1, output_numpy=True):
        mr_data.append(x)
    count = 0
    for x in tf_data:
        for k, v in x.items():
            if isinstance(v, np.ndarray):
                assert (v == mr_data[count][cast_name(k)]).all()
            else:
                assert v == mr_data[count][cast_name(k)]
        count += 1
    assert count == 10

    if os.path.exists("{}".format(AUTO_FILE)):
        os.remove("{}".format(AUTO_FILE))
    if os.path.exists("{}.db".format(AUTO_FILE)):
        os.remove("{}.db".format(AUTO_FILE))


def generator_dynamic_1d():
    arr = []
    for i in range(10):
        if i % 5 == 0:
            arr = []
        arr += [i]
        yield (np.array(arr),)


def generator_dynamic_2d_0():
    for i in range(10):
        if i < 5:
            yield (np.arange(5).reshape([1, 5]),)
        else:
            yield (np.arange(10).reshape([2, 5]),)


def generator_dynamic_2d_1():
    for i in range(10):
        if i < 5:
            yield (np.arange(5).reshape([5, 1]),)
        else:
            yield (np.arange(10).reshape([5, 2]),)


def test_case_08(add_remove_file):

    # apply dataset operations
    d1 = ds.GeneratorDataset(generator_dynamic_1d, ["data"], shuffle=False)

    d1.save(AUTO_FILE)

    d2 = ds.MindDataset(dataset_file=AUTO_FILE,
                        num_parallel_workers=num_readers,
                        shuffle=False)

    i = 0
    arr = []
    for item in d2.create_dict_iterator(num_epochs=1, output_numpy=True):
        if i % 5 == 0:
            arr = []
        arr += [i]
        golden = np.array(arr)
        np.testing.assert_array_equal(item["data"], golden)
        i = i + 1


def test_case_09(add_remove_file):

    # apply dataset operations
    d1 = ds.GeneratorDataset(generator_dynamic_2d_0, ["data"], shuffle=False)

    d1.save(AUTO_FILE)

    d2 = ds.MindDataset(dataset_file=AUTO_FILE,
                        num_parallel_workers=num_readers,
                        shuffle=False)

    i = 0
    for item in d2.create_dict_iterator(num_epochs=1, output_numpy=True):
        if i < 5:
            golden = np.arange(5).reshape([1, 5])
        else:
            golden = np.arange(10).reshape([2, 5])
        np.testing.assert_array_equal(item["data"], golden)
        i = i + 1


def test_case_10(add_remove_file):

    # apply dataset operations
    d1 = ds.GeneratorDataset(generator_dynamic_2d_1, ["data"], shuffle=False)

    with pytest.raises(Exception, match=
                       "Error: besides dimension 0, other dimension shape is different from the previous's"):
        d1.save(AUTO_FILE)
