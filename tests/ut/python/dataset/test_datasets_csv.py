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
import numpy as np
import pytest
import mindspore.dataset as ds

DATA_FILE = '../data/dataset/testCSV/1.csv'


def test_csv_dataset_basic():
    """
    Test CSV with repeat, skip and so on
    """
    TRAIN_FILE = '../data/dataset/testCSV/1.csv'

    buffer = []
    data = ds.CSVDataset(
        TRAIN_FILE,
        field_delim=',',
        column_defaults=["0", 0, 0.0, "0"],
        column_names=['1', '2', '3', '4'],
        shuffle=False)
    data = data.repeat(2)
    data = data.skip(2)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append(d)
    assert len(buffer) == 4


def test_csv_dataset_one_file():
    data = ds.CSVDataset(
        DATA_FILE,
        column_defaults=["1", "2", "3", "4"],
        column_names=['col1', 'col2', 'col3', 'col4'],
        shuffle=False)
    buffer = []
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append(d)
    assert len(buffer) == 3


def test_csv_dataset_all_file():
    APPEND_FILE = '../data/dataset/testCSV/2.csv'
    data = ds.CSVDataset(
        [DATA_FILE, APPEND_FILE],
        column_defaults=["1", "2", "3", "4"],
        column_names=['col1', 'col2', 'col3', 'col4'],
        shuffle=False)
    buffer = []
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append(d)
    assert len(buffer) == 10


def test_csv_dataset_num_samples():
    data = ds.CSVDataset(
        DATA_FILE,
        column_defaults=["1", "2", "3", "4"],
        column_names=['col1', 'col2', 'col3', 'col4'],
        shuffle=False, num_samples=2)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 2


def test_csv_dataset_distribution():
    TEST_FILE = '../data/dataset/testCSV/1.csv'
    data = ds.CSVDataset(
        TEST_FILE,
        column_defaults=["1", "2", "3", "4"],
        column_names=['col1', 'col2', 'col3', 'col4'],
        shuffle=False, num_shards=2, shard_id=0)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 2


def test_csv_dataset_quoted():
    TEST_FILE = '../data/dataset/testCSV/quoted.csv'
    data = ds.CSVDataset(
        TEST_FILE,
        column_defaults=["", "", "", ""],
        column_names=['col1', 'col2', 'col3', 'col4'],
        shuffle=False)
    buffer = []
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.extend([d['col1'].item().decode("utf8"),
                       d['col2'].item().decode("utf8"),
                       d['col3'].item().decode("utf8"),
                       d['col4'].item().decode("utf8")])
    assert buffer == ['a', 'b', 'c', 'd']


def test_csv_dataset_separated():
    TEST_FILE = '../data/dataset/testCSV/separated.csv'
    data = ds.CSVDataset(
        TEST_FILE,
        field_delim='|',
        column_defaults=["", "", "", ""],
        column_names=['col1', 'col2', 'col3', 'col4'],
        shuffle=False)
    buffer = []
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.extend([d['col1'].item().decode("utf8"),
                       d['col2'].item().decode("utf8"),
                       d['col3'].item().decode("utf8"),
                       d['col4'].item().decode("utf8")])
    assert buffer == ['a', 'b', 'c', 'd']


def test_csv_dataset_embedded():
    TEST_FILE = '../data/dataset/testCSV/embedded.csv'
    data = ds.CSVDataset(
        TEST_FILE,
        column_defaults=["", "", "", ""],
        column_names=['col1', 'col2', 'col3', 'col4'],
        shuffle=False)
    buffer = []
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.extend([d['col1'].item().decode("utf8"),
                       d['col2'].item().decode("utf8"),
                       d['col3'].item().decode("utf8"),
                       d['col4'].item().decode("utf8")])
    assert buffer == ['a,b', 'c"d', 'e\nf', ' g ']


def test_csv_dataset_chinese():
    TEST_FILE = '../data/dataset/testCSV/chinese.csv'
    data = ds.CSVDataset(
        TEST_FILE,
        column_defaults=["", "", "", "", ""],
        column_names=['col1', 'col2', 'col3', 'col4', 'col5'],
        shuffle=False)
    buffer = []
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.extend([d['col1'].item().decode("utf8"),
                       d['col2'].item().decode("utf8"),
                       d['col3'].item().decode("utf8"),
                       d['col4'].item().decode("utf8"),
                       d['col5'].item().decode("utf8")])
    assert buffer == ['大家', '早上好', '中午好', '下午好', '晚上好']


def test_csv_dataset_header():
    TEST_FILE = '../data/dataset/testCSV/header.csv'
    data = ds.CSVDataset(
        TEST_FILE,
        column_defaults=["", "", "", ""],
        shuffle=False)
    buffer = []
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.extend([d['col1'].item().decode("utf8"),
                       d['col2'].item().decode("utf8"),
                       d['col3'].item().decode("utf8"),
                       d['col4'].item().decode("utf8")])
    assert buffer == ['a', 'b', 'c', 'd']


def test_csv_dataset_number():
    TEST_FILE = '../data/dataset/testCSV/number.csv'
    data = ds.CSVDataset(
        TEST_FILE,
        column_defaults=[0.0, 0.0, 0, 0.0],
        column_names=['col1', 'col2', 'col3', 'col4'],
        shuffle=False)
    buffer = []
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.extend([d['col1'].item(),
                       d['col2'].item(),
                       d['col3'].item(),
                       d['col4'].item()])
    assert np.allclose(buffer, [3.0, 0.3, 4, 55.5])


def test_csv_dataset_field_delim_none():
    """
    Test CSV with field_delim=None
    """
    TRAIN_FILE = '../data/dataset/testCSV/1.csv'

    buffer = []
    data = ds.CSVDataset(
        TRAIN_FILE,
        field_delim=None,
        column_defaults=["0", 0, 0.0, "0"],
        column_names=['1', '2', '3', '4'],
        shuffle=False)
    data = data.repeat(2)
    data = data.skip(2)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append(d)
    assert len(buffer) == 4


def test_csv_dataset_size():
    TEST_FILE = '../data/dataset/testCSV/size.csv'
    data = ds.CSVDataset(
        TEST_FILE,
        column_defaults=[0.0, 0.0, 0, 0.0],
        column_names=['col1', 'col2', 'col3', 'col4'],
        shuffle=False)
    assert data.get_dataset_size() == 5


def test_csv_dataset_type_error():
    TEST_FILE = '../data/dataset/testCSV/exception.csv'
    data = ds.CSVDataset(
        TEST_FILE,
        column_defaults=["", 0, "", ""],
        column_names=['col1', 'col2', 'col3', 'col4'],
        shuffle=False)
    with pytest.raises(Exception) as err:
        for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass
    assert "type does not match" in str(err.value)


def test_csv_dataset_exception():
    TEST_FILE = '../data/dataset/testCSV/exception.csv'
    data = ds.CSVDataset(
        TEST_FILE,
        column_defaults=["", "", "", ""],
        column_names=['col1', 'col2', 'col3', 'col4'],
        shuffle=False)
    with pytest.raises(Exception) as err:
        for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass
    assert "failed to parse file" in str(err.value)

    TEST_FILE1 = '../data/dataset/testCSV/quoted.csv'
    def exception_func(item):
        raise Exception("Error occur!")

    try:
        data = ds.CSVDataset(
            TEST_FILE1,
            column_defaults=["", "", "", ""],
            column_names=['col1', 'col2', 'col3', 'col4'],
            shuffle=False)
        data = data.map(operations=exception_func, input_columns=["col1"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data = ds.CSVDataset(
            TEST_FILE1,
            column_defaults=["", "", "", ""],
            column_names=['col1', 'col2', 'col3', 'col4'],
            shuffle=False)
        data = data.map(operations=exception_func, input_columns=["col2"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data = ds.CSVDataset(
            TEST_FILE1,
            column_defaults=["", "", "", ""],
            column_names=['col1', 'col2', 'col3', 'col4'],
            shuffle=False)
        data = data.map(operations=exception_func, input_columns=["col3"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data = ds.CSVDataset(
            TEST_FILE1,
            column_defaults=["", "", "", ""],
            column_names=['col1', 'col2', 'col3', 'col4'],
            shuffle=False)
        data = data.map(operations=exception_func, input_columns=["col4"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)


def test_csv_dataset_duplicate_columns():
    data = ds.CSVDataset(
        DATA_FILE,
        column_defaults=["1", "2", "3", "4"],
        column_names=['col1', 'col2', 'col3', 'col4', 'col1', 'col2', 'col3', 'col4'],
        shuffle=False)
    with pytest.raises(RuntimeError) as info:
        _ = data.create_dict_iterator(num_epochs=1, output_numpy=True)
    assert "Invalid parameter, duplicate column names are not allowed: col1" in str(info.value)
    assert "column_names" in str(info.value)


if __name__ == "__main__":
    test_csv_dataset_basic()
    test_csv_dataset_one_file()
    test_csv_dataset_all_file()
    test_csv_dataset_num_samples()
    test_csv_dataset_distribution()
    test_csv_dataset_quoted()
    test_csv_dataset_separated()
    test_csv_dataset_embedded()
    test_csv_dataset_chinese()
    test_csv_dataset_header()
    test_csv_dataset_number()
    test_csv_dataset_field_delim_none()
    test_csv_dataset_size()
    test_csv_dataset_type_error()
    test_csv_dataset_exception()
    test_csv_dataset_duplicate_columns()
