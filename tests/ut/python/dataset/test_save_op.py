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
This is the test module for saveOp.
"""
import os
from string import punctuation

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore import dtype as mstype
import mindspore.dataset.transforms as transforms
from mindspore import log as logger
from mindspore.mindrecord import FileWriter, set_enc_key, set_enc_mode, set_hash_mode

TFRECORD_FILES = "../data/mindrecord/testTFRecordData/dummy.tfrecord"
FILES_NUM = 1
num_readers = 1


def remove_file(file_name):
    """add/remove cv file"""
    if os.path.exists("{}".format(file_name)):
        os.remove("{}".format(file_name))
    if os.path.exists("{}.db".format(file_name)):
        os.remove("{}.db".format(file_name))


def test_case_00():
    """
    Feature: Save op
    Description: All bin data
    Expectation: Generated mindrecord file
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data = [{"image1": bytes("image1 bytes abcddddd", encoding='UTF-8'),
             "image2": bytes("image1 bytes def", encoding='UTF-8'),
             "image3": bytes("image1 bytes ghixxxxxxxxxx", encoding='UTF-8'),
             "image4": bytes("image1 bytes jklzz", encoding='UTF-8'),
             "image5": bytes("image1 bytes mno", encoding='UTF-8')},
            {"image1": bytes("image2 bytes abca", encoding='UTF-8'),
             "image2": bytes("image2 bytes defbb", encoding='UTF-8'),
             "image3": bytes("image2 bytes ghiccc", encoding='UTF-8'),
             "image4": bytes("image2 bytes jkldddd", encoding='UTF-8'),
             "image5": bytes("image2 bytes mnoeeeeeee", encoding='UTF-8')},
            {"image1": bytes("image3 bytes abciiiiiiiiiiii", encoding='UTF-8'),
             "image2": bytes("image3 bytes def", encoding='UTF-8'),
             "image3": bytes("image3 bytes ghiooooo", encoding='UTF-8'),
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
    writer = FileWriter(file_name, FILES_NUM)
    writer.add_schema(schema, "schema")
    writer.write_raw_data(data)
    writer.commit()

    file_name_auto = './'
    file_name_auto += os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    file_name_auto += '_auto'
    d1 = ds.MindDataset(file_name, None, num_readers, shuffle=False)
    d1.save(file_name_auto, FILES_NUM)

    d2 = ds.MindDataset(dataset_files=file_name_auto,
                        num_parallel_workers=num_readers,
                        shuffle=False)
    assert d2.get_dataset_size() == 5
    num_iter = 0
    for item in d2.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 5
        for field in item:
            assert (item[field] == data[num_iter][field]).all()
        num_iter += 1
    assert num_iter == 5
    remove_file(file_name)
    remove_file(file_name_auto)

    file_name_auto = './'
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data = [{"file_name": "001.jpg", "label": 43},
            {"file_name": "002.jpg", "label": 91},
            {"file_name": "003.jpg", "label": 61},
            {"file_name": "004.jpg", "label": 29},
            {"file_name": "005.jpg", "label": 78},
            {"file_name": "006.jpg", "label": 37}]
    schema = {"file_name": {"type": "string"},
              "label": {"type": "int32"}
              }

    writer = FileWriter(file_name, FILES_NUM)
    writer.add_schema(schema, "schema")
    writer.write_raw_data(data)
    writer.commit()

    file_name_auto = './'
    file_name_auto += os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    file_name_auto += '_auto'
    d1 = ds.MindDataset(file_name, None, num_readers, shuffle=False)
    d1.save(file_name_auto, FILES_NUM)

    d2 = ds.MindDataset(dataset_files=file_name_auto,
                        num_parallel_workers=num_readers,
                        shuffle=False)
    assert d2.get_dataset_size() == 6
    num_iter = 0
    for item in d2.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(item)
        assert len(item) == 2
        for field in item:
            assert (item[field] == data[num_iter][field]).all()
        num_iter += 1
    assert num_iter == 6
    remove_file(file_name)
    remove_file(file_name_auto)


def test_case_02():  # muti-bytes
    """
    Feature: Save op
    Description: Multiple byte fields
    Expectation: Generated mindrecord file
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
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
    writer = FileWriter(file_name, FILES_NUM)
    writer.add_schema(schema, "schema")
    writer.write_raw_data(data)
    writer.commit()

    file_name_auto = './'
    file_name_auto += os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    file_name_auto += '_auto'
    d1 = ds.MindDataset(file_name, None, num_readers, shuffle=False)
    d1.save(file_name_auto, FILES_NUM)

    d2 = ds.MindDataset(dataset_files=file_name_auto,
                        num_parallel_workers=num_readers,
                        shuffle=False)
    assert d2.get_dataset_size() == 6
    num_iter = 0
    for item in d2.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 13
        for field in item:
            if isinstance(item[field], np.ndarray):
                if item[field].dtype == np.float32:
                    assert (item[field] == np.array(data[num_iter][field], np.float32)).all()
                else:
                    assert (item[field] == data[num_iter][field]).all()
            else:
                assert item[field] == data[num_iter][field]
        num_iter += 1
    assert num_iter == 6
    remove_file(file_name)
    remove_file(file_name_auto)


def generator_1d():
    for i in range(10):
        yield (np.array([i]),)


def test_case_03():
    """
    Feature: Save op
    Description: 1D numpy array
    Expectation: Generated mindrecord file
    """
    file_name_auto = './'
    file_name_auto += os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    file_name_auto += '_auto'
    # apply dataset operations
    d1 = ds.GeneratorDataset(generator_1d, ["data"], shuffle=False)

    d1.save(file_name_auto)

    d2 = ds.MindDataset(dataset_files=file_name_auto,
                        num_parallel_workers=num_readers,
                        shuffle=False)

    i = 0
    # each data is a dictionary
    for item in d2.create_dict_iterator(num_epochs=1, output_numpy=True):
        golden = np.array([i])
        np.testing.assert_array_equal(item["data"], golden)
        i = i + 1
    remove_file(file_name_auto)


def generator_with_type(t):
    for i in range(64):
        yield (np.array([i], dtype=t),)


def type_tester(t):
    file_name_auto = './'
    file_name_auto += os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    file_name_auto += '_auto'
    logger.info("Test with Type {}".format(t.__name__))

    # apply dataset operations
    data1 = ds.GeneratorDataset((lambda: generator_with_type(t)), ["data"], shuffle=False)

    data1 = data1.batch(4)

    data1 = data1.repeat(3)

    data1.save(file_name_auto)

    d2 = ds.MindDataset(dataset_files=file_name_auto,
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
    remove_file(file_name_auto)


def test_case_04():
    # uint8 will drop shape as mindrecord store uint8 as bytes
    types = [np.int8, np.int16, np.int32, np.int64,
             np.uint16, np.uint32, np.float32, np.float64]

    for t in types:
        type_tester(t)


def test_case_05():
    """
    Feature: Save op
    Description: Exception Test
    Expectation: Exception
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    d1 = ds.GeneratorDataset(generator_1d, ["data"], shuffle=False)

    with pytest.raises(Exception, match="num_files should between 0 and 1000."):
        d1.save(file_name, 0)


def test_case_06():
    """
    Feature: Save op
    Description: Exception Test
    Expectation: Exception
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    d1 = ds.GeneratorDataset(generator_1d, ["data"], shuffle=False)

    with pytest.raises(Exception, match="tfrecord dataset format is not supported."):
        d1.save(file_name, 1, "tfrecord")


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
    """
    Feature: Save op
    Description: Save tfrecord files
    Expectation: Generated mindrecord file
    """
    file_name_auto = './'
    file_name_auto += os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    file_name_auto += '_auto'
    d1 = ds.TFRecordDataset(TFRECORD_FILES, shuffle=False)
    tf_data = []
    for x in d1.create_dict_iterator(num_epochs=1, output_numpy=True):
        tf_data.append(x)
    d1.save(file_name_auto, FILES_NUM)
    d2 = ds.MindDataset(dataset_files=file_name_auto,
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
    remove_file(file_name_auto)


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


def test_case_08():
    """
    Feature: Save op
    Description: Save dynamic 1D numpy array
    Expectation: Generated mindrecord file
    """
    file_name_auto = './'
    file_name_auto += os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    file_name_auto += '_auto'
    # apply dataset operations
    d1 = ds.GeneratorDataset(generator_dynamic_1d, ["data"], shuffle=False)

    d1.save(file_name_auto)

    d2 = ds.MindDataset(dataset_files=file_name_auto,
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
    remove_file(file_name_auto)


def test_case_09():
    """
    Feature: Save op
    Description: Save dynamic 2D numpy array
    Expectation: Generated mindrecord file
    """
    file_name_auto = './'
    file_name_auto += os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    file_name_auto += '_auto'
    # apply dataset operations
    d1 = ds.GeneratorDataset(generator_dynamic_2d_0, ["data"], shuffle=False)

    d1.save(file_name_auto)

    d2 = ds.MindDataset(dataset_files=file_name_auto,
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
    remove_file(file_name_auto)


def test_case_10():
    """
    Feature: Save op
    Description: Save 2D Tensor of different shape
    Expectation: Exception
    """
    file_name_auto = './'
    file_name_auto += os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    file_name_auto += '_auto'

    # apply dataset operations
    d1 = ds.GeneratorDataset(generator_dynamic_2d_1, ["data"], shuffle=False)

    with pytest.raises(Exception,
                       match="Tensor with dynamic shape do not currently support saving. "
                             "Except for the shape of dimension 0, the other dimension shapes must be fixed. "
                             "You can reshape the Tensor to a fixed shape before saving."):
        d1.save(file_name_auto)
    remove_file(file_name_auto)


def test_case_all_types():
    """
    Feature: Save op
    Description: Test converting datasets of various data types to MindRecord
    Expectation: Data read from the saved MindRecord is still of the original data type
    """

    dumpy_data = [{"bool": True, "int8": 4, "uint8": 255, "int16": 1000, "uint16": 9999, "int32": 12345,
                   "uint32": 123450, "int64": 6434567890, "uint64": 116434567890, "float16": 12.123,
                   "float32": 1.6875923, "float64": 123456.98765, "string": "it's so cool.",
                   "bytes": bytes("image1 bytes abc", encoding='UTF-8'),
                   "int8_array": np.array([1, 2, 3], dtype=np.int8),
                   "uint8_array": np.array([255, 255, 255], dtype=np.uint8),
                   "int16_array": np.array([1000, 1000, 1000], dtype=np.int16),
                   "uint16_array": np.array([9999, 9999, 9999], dtype=np.uint16),
                   "int32_array": np.array([1000, 12345, 1000], dtype=np.int32),
                   "uint32_array": np.array([123450, 1000, 1000], dtype=np.uint32),
                   "int64_array": np.array([6434567890, 12345, 1000], dtype=np.int64),
                   "uint64_array": np.array([1000, 12345, 116434567890], dtype=np.uint64),
                   "float16_array": np.array([12.123, 123.79, 10.0], dtype=np.float16),
                   "float32_array": np.array([1.6875923, 1245.361, 1000.9821], dtype=np.float32),
                   "float64_array": np.array([123456.98765, 12345.0, 1000.123456], dtype=np.float64)},
                  {"bool": False, "int8": 0, "uint8": 128, "int16": 999, "uint16": 9669, "int32": 312345,
                   "uint32": 9123450, "int64": 3434567890, "uint64": 316434567890, "float16": 12.223,
                   "float32": 1.9875923, "float64": 223456.98765, "string": "it's so cool2.",
                   "bytes": bytes("image2 bytes abczzzzz", encoding='UTF-8'),
                   "int8_array": np.array([3, 2, 3], dtype=np.int8),
                   "uint8_array": np.array([255, 1, 255], dtype=np.uint8),
                   "int16_array": np.array([1000, 100, 1000], dtype=np.int16),
                   "uint16_array": np.array([9999, 999, 9999], dtype=np.uint16),
                   "int32_array": np.array([100, 12345, 1000], dtype=np.int32),
                   "uint32_array": np.array([123450, 1000, 100], dtype=np.uint32),
                   "int64_array": np.array([6434567890, 2345, 1000], dtype=np.int64),
                   "uint64_array": np.array([1000, 12345, 316434567890], dtype=np.uint64),
                   "float16_array": np.array([12.923, 123.79, 10.0], dtype=np.float16),
                   "float32_array": np.array([1.6875923, 1245.961, 1000.9821], dtype=np.float32),
                   "float64_array": np.array([123456.98765, 12345.0, 1000.93456], dtype=np.float64)}]

    class Iterable:
        def __init__(self, data):
            self._datas = data

        def __getitem__(self, index):
            return self._datas[index]["bool"], self._datas[index]["int8"], self._datas[index]["uint8"], \
                self._datas[index]["int16"], self._datas[index]["uint16"], self._datas[index]["int32"], \
                self._datas[index]["uint32"], self._datas[index]["int64"], self._datas[index]["uint64"], \
                self._datas[index]["float16"], self._datas[index]["float32"], self._datas[index]["float64"], \
                self._datas[index]["string"], self._datas[index]["bytes"], self._datas[index]["int8_array"], \
                self._datas[index]["uint8_array"], self._datas[index]["int16_array"], \
                self._datas[index]["uint16_array"], self._datas[index]["int32_array"], \
                self._datas[index]["uint32_array"], self._datas[index]["int64_array"], \
                self._datas[index]["uint64_array"], self._datas[index]["float16_array"], \
                self._datas[index]["float32_array"], self._datas[index]["float64_array"]

        def __len__(self):
            return len(self._datas)

    data_source = Iterable(dumpy_data)
    dataset = ds.GeneratorDataset(source=data_source, column_names=["bool", "int8", "uint8", "int16", "uint16",
                                                                    "int32", "uint32", "int64", "uint64",
                                                                    "float16", "float32", "float64", "string",
                                                                    "bytes", "int8_array", "uint8_array",
                                                                    "int16_array", "uint16_array", "int32_array",
                                                                    "uint32_array", "int64_array", "uint64_array",
                                                                    "float16_array", "float32_array", "float64_array"],
                                  shuffle=False)
    dataset = dataset.map(operations=transforms.TypeCast(mstype.bool_), input_columns="bool")
    dataset = dataset.map(operations=transforms.TypeCast(mstype.int8), input_columns="int8")
    dataset = dataset.map(operations=transforms.TypeCast(mstype.uint8), input_columns="uint8")
    dataset = dataset.map(operations=transforms.TypeCast(mstype.int16), input_columns="int16")
    dataset = dataset.map(operations=transforms.TypeCast(mstype.uint16), input_columns="uint16")
    dataset = dataset.map(operations=transforms.TypeCast(mstype.int32), input_columns="int32")
    dataset = dataset.map(operations=transforms.TypeCast(mstype.uint32), input_columns="uint32")
    dataset = dataset.map(operations=transforms.TypeCast(mstype.int64), input_columns="int64")
    dataset = dataset.map(operations=transforms.TypeCast(mstype.uint64), input_columns="uint64")
    dataset = dataset.map(operations=transforms.TypeCast(mstype.float16), input_columns="float16")
    dataset = dataset.map(operations=transforms.TypeCast(mstype.float32), input_columns="float32")
    dataset = dataset.map(operations=transforms.TypeCast(mstype.float64), input_columns="float64")

    mindrecord_filename = "test_all_types.mindrecord"
    remove_file(mindrecord_filename)
    dataset.save(mindrecord_filename)

    data_set = ds.MindDataset(dataset_files=mindrecord_filename, shuffle=False)
    count = 0
    for item in data_set.create_dict_iterator(output_numpy=True, num_epochs=1):
        bool_value = 0
        if dumpy_data[count]["bool"]:
            bool_value = 1
        assert bool_value == item["bool"]
        assert dumpy_data[count]["int8"] == item["int8"]
        assert dumpy_data[count]["uint8"] == item["uint8"]
        assert dumpy_data[count]["int16"] == item["int16"]
        assert dumpy_data[count]["uint16"] == item["uint16"]
        assert dumpy_data[count]["int32"] == item["int32"]
        assert dumpy_data[count]["uint32"] == item["uint32"]
        assert dumpy_data[count]["int64"] == item["int64"]
        assert dumpy_data[count]["uint64"] == item["uint64"]
        assert np.allclose(np.array(dumpy_data[count]["float16"], dtype=np.float32), item["float16"], 0.001, 0.001)
        assert np.allclose(np.array(dumpy_data[count]["float32"], dtype=np.float32), item["float32"], 0.0001, 0.0001)
        assert np.allclose(np.array(dumpy_data[count]["float64"], dtype=np.float32), item["float64"], 0.0001, 0.0001)
        assert dumpy_data[count]["string"] == item["string"]
        assert dumpy_data[count]["bytes"] == item["bytes"]
        assert (dumpy_data[count]["int8_array"] == item["int8_array"]).all()
        assert (dumpy_data[count]["uint8_array"] == item["uint8_array"]).all()
        assert (dumpy_data[count]["int16_array"] == item["int16_array"]).all()
        assert (dumpy_data[count]["uint16_array"] == item["uint16_array"]).all()
        assert (dumpy_data[count]["int32_array"] == item["int32_array"]).all()
        assert (dumpy_data[count]["uint32_array"] == item["uint32_array"]).all()
        assert (dumpy_data[count]["int64_array"] == item["int64_array"]).all()
        assert (dumpy_data[count]["uint64_array"] == item["uint64_array"]).all()
        assert np.allclose(dumpy_data[count]["float16_array"], item["float16_array"], 0.001, 0.001)
        assert np.allclose(dumpy_data[count]["float32_array"], item["float32_array"], 0.0001, 0.0001)
        assert np.allclose(dumpy_data[count]["float64_array"], item["float64_array"], 0.0001, 0.0001)
        count += 1
    assert count == 2

    remove_file(mindrecord_filename)


def save_with_encode_and_hash_check(file_name, enc_key, enc_mode, hash_mode):
    """Save with encode and hash check"""

    ## single file
    if os.path.exists(file_name):
        os.remove("{}".format(file_name))
    if os.path.exists(file_name + ".db"):
        os.remove("{}".format(file_name + ".db"))

    set_enc_key(enc_key)
    set_enc_mode(enc_mode)
    set_hash_mode(hash_mode)

    d1 = ds.GeneratorDataset(generator_1d, ["data"], shuffle=False)
    d1.save(file_name)

    dataset = ds.MindDataset(dataset_files=[file_name])
    assert dataset.get_dataset_size() == 10

    ## single file
    if os.path.exists(file_name):
        os.remove("{}".format(file_name))
    if os.path.exists(file_name + ".db"):
        os.remove("{}".format(file_name + ".db"))

    set_enc_key(None)
    set_hash_mode(None)


def test_case_with_encode_and_hash_check():
    """
    Feature: Save op
    Description: Save with encode and hash check
    Expectation: Success
    """
    file_name = './'
    file_name += os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    save_with_encode_and_hash_check(file_name, None, "AES-CBC", None)
    save_with_encode_and_hash_check(file_name, "abcdefghijklmnop01234567", "AES-CBC", None)
    save_with_encode_and_hash_check(file_name, None, "AES-CBC", "sha3_384")
    save_with_encode_and_hash_check(file_name, "89012345abcdefgh", "SM4-CBC", "sha512")
