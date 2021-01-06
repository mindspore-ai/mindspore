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
"""
Test TFRecordDataset Ops
"""
import numpy as np
import pytest

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
from mindspore import log as logger
from util import save_and_check_dict

FILES = ["../data/dataset/testTFTestAllTypes/test.data"]
DATASET_ROOT = "../data/dataset/testTFTestAllTypes/"
SCHEMA_FILE = "../data/dataset/testTFTestAllTypes/datasetSchema.json"
DATA_FILES2 = ["../data/dataset/test_tf_file_3_images2/train-0000-of-0001.data",
               "../data/dataset/test_tf_file_3_images2/train-0000-of-0002.data",
               "../data/dataset/test_tf_file_3_images2/train-0000-of-0003.data",
               "../data/dataset/test_tf_file_3_images2/train-0000-of-0004.data"]
SCHEMA_FILE2 = "../data/dataset/test_tf_file_3_images2/datasetSchema.json"
GENERATE_GOLDEN = False


def test_tfrecord_shape():
    logger.info("test_tfrecord_shape")
    schema_file = "../data/dataset/testTFTestAllTypes/datasetSchemaRank0.json"
    ds1 = ds.TFRecordDataset(FILES, schema_file)
    ds1 = ds1.batch(2)
    for data in ds1.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(data)
    output_shape = ds1.output_shapes()
    assert len(output_shape[-1]) == 1


def test_tfrecord_read_all_dataset():
    logger.info("test_tfrecord_read_all_dataset")
    schema_file = "../data/dataset/testTFTestAllTypes/datasetSchemaNoRow.json"
    ds1 = ds.TFRecordDataset(FILES, schema_file)
    assert ds1.get_dataset_size() == 12
    count = 0
    for _ in ds1.create_tuple_iterator(num_epochs=1):
        count += 1
    assert count == 12


def test_tfrecord_num_samples():
    logger.info("test_tfrecord_num_samples")
    schema_file = "../data/dataset/testTFTestAllTypes/datasetSchema7Rows.json"
    ds1 = ds.TFRecordDataset(FILES, schema_file, num_samples=8)
    assert ds1.get_dataset_size() == 8
    count = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        count += 1
    assert count == 8


def test_tfrecord_num_samples2():
    logger.info("test_tfrecord_num_samples2")
    schema_file = "../data/dataset/testTFTestAllTypes/datasetSchema7Rows.json"
    ds1 = ds.TFRecordDataset(FILES, schema_file)
    assert ds1.get_dataset_size() == 7
    count = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        count += 1
    assert count == 7


def test_tfrecord_shape2():
    logger.info("test_tfrecord_shape2")
    ds1 = ds.TFRecordDataset(FILES, SCHEMA_FILE)
    ds1 = ds1.batch(2)
    output_shape = ds1.output_shapes()
    assert len(output_shape[-1]) == 2


def test_tfrecord_files_basic():
    logger.info("test_tfrecord_files_basic")

    data = ds.TFRecordDataset(FILES, SCHEMA_FILE, shuffle=ds.Shuffle.FILES)
    filename = "tfrecord_files_basic.npz"
    save_and_check_dict(data, filename, generate_golden=GENERATE_GOLDEN)


def test_tfrecord_no_schema():
    logger.info("test_tfrecord_no_schema")

    data = ds.TFRecordDataset(FILES, shuffle=ds.Shuffle.FILES)
    filename = "tfrecord_no_schema.npz"
    save_and_check_dict(data, filename, generate_golden=GENERATE_GOLDEN)


def test_tfrecord_pad():
    logger.info("test_tfrecord_pad")

    schema_file = "../data/dataset/testTFTestAllTypes/datasetSchemaPadBytes10.json"
    data = ds.TFRecordDataset(FILES, schema_file, shuffle=ds.Shuffle.FILES)
    filename = "tfrecord_pad_bytes10.npz"
    save_and_check_dict(data, filename, generate_golden=GENERATE_GOLDEN)


def test_tfrecord_read_files():
    logger.info("test_tfrecord_read_files")
    pattern = DATASET_ROOT + "/test.data"
    data = ds.TFRecordDataset(pattern, SCHEMA_FILE, shuffle=ds.Shuffle.FILES)
    assert sum([1 for _ in data]) == 12

    pattern = DATASET_ROOT + "/test2.data"
    data = ds.TFRecordDataset(pattern, SCHEMA_FILE, shuffle=ds.Shuffle.FILES)
    assert sum([1 for _ in data]) == 12

    pattern = DATASET_ROOT + "/*.data"
    data = ds.TFRecordDataset(pattern, SCHEMA_FILE, num_samples=24, shuffle=ds.Shuffle.FILES)
    assert sum([1 for _ in data]) == 24

    pattern = DATASET_ROOT + "/*.data"
    data = ds.TFRecordDataset(pattern, SCHEMA_FILE, num_samples=3, shuffle=ds.Shuffle.FILES)
    assert sum([1 for _ in data]) == 3

    data = ds.TFRecordDataset([DATASET_ROOT + "/test.data", DATASET_ROOT + "/test2.data"],
                              SCHEMA_FILE, num_samples=24, shuffle=ds.Shuffle.FILES)
    assert sum([1 for _ in data]) == 24


def test_tfrecord_multi_files():
    logger.info("test_tfrecord_multi_files")
    data1 = ds.TFRecordDataset(DATA_FILES2, SCHEMA_FILE2, shuffle=False)
    data1 = data1.repeat(1)
    num_iter = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_iter += 1

    assert num_iter == 12


def test_tfrecord_schema():
    logger.info("test_tfrecord_schema")
    schema = ds.Schema()
    schema.add_column('col_1d', de_type=mstype.int64, shape=[2])
    schema.add_column('col_2d', de_type=mstype.int64, shape=[2, 2])
    schema.add_column('col_3d', de_type=mstype.int64, shape=[2, 2, 2])
    schema.add_column('col_binary', de_type=mstype.uint8, shape=[1])
    schema.add_column('col_float', de_type=mstype.float32, shape=[1])
    schema.add_column('col_sint16', de_type=mstype.int64, shape=[1])
    schema.add_column('col_sint32', de_type=mstype.int64, shape=[1])
    schema.add_column('col_sint64', de_type=mstype.int64, shape=[1])
    data1 = ds.TFRecordDataset(FILES, schema=schema, shuffle=ds.Shuffle.FILES)

    data2 = ds.TFRecordDataset(FILES, schema=SCHEMA_FILE, shuffle=ds.Shuffle.FILES)

    for d1, d2 in zip(data1, data2):
        for t1, t2 in zip(d1, d2):
            np.testing.assert_array_equal(t1.asnumpy(), t2.asnumpy())


def test_tfrecord_shuffle():
    logger.info("test_tfrecord_shuffle")
    ds.config.set_seed(1)
    data1 = ds.TFRecordDataset(FILES, schema=SCHEMA_FILE, shuffle=ds.Shuffle.GLOBAL)
    data2 = ds.TFRecordDataset(FILES, schema=SCHEMA_FILE, shuffle=ds.Shuffle.FILES)
    data2 = data2.shuffle(10000)

    for d1, d2 in zip(data1, data2):
        for t1, t2 in zip(d1, d2):
            np.testing.assert_array_equal(t1.asnumpy(), t2.asnumpy())


def test_tfrecord_shard():
    logger.info("test_tfrecord_shard")
    tf_files = ["../data/dataset/tf_file_dataset/test1.data", "../data/dataset/tf_file_dataset/test2.data",
                "../data/dataset/tf_file_dataset/test3.data", "../data/dataset/tf_file_dataset/test4.data"]

    def get_res(shard_id, num_repeats):
        data1 = ds.TFRecordDataset(tf_files, num_shards=2, shard_id=shard_id, num_samples=3,
                                   shuffle=ds.Shuffle.FILES)
        data1 = data1.repeat(num_repeats)
        res = list()
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            res.append(item["scalars"][0])
        return res

    # get separate results from two workers. the 2 results need to satisfy 2 criteria
    # 1. two workers always give different results in same epoch (e.g. wrkr1:f1&f3, wrkr2:f2&f4  in one epoch)
    # 2. with enough epochs, both workers will get the entire dataset (e,g. ep1_wrkr1: f1&f3, ep2,_wrkr1 f2&f4)
    worker1_res = get_res(0, 16)
    worker2_res = get_res(1, 16)
    # Confirm each worker gets 3x16=48 rows
    assert len(worker1_res) == 48
    assert len(worker1_res) == len(worker2_res)
    # check criteria 1
    for i, _ in enumerate(worker1_res):
        assert worker1_res[i] != worker2_res[i]
    # check criteria 2
    assert set(worker2_res) == set(worker1_res)


def test_tfrecord_shard_equal_rows():
    logger.info("test_tfrecord_shard_equal_rows")
    tf_files = ["../data/dataset/tf_file_dataset/test1.data", "../data/dataset/tf_file_dataset/test2.data",
                "../data/dataset/tf_file_dataset/test3.data", "../data/dataset/tf_file_dataset/test4.data"]

    def get_res(num_shards, shard_id, num_repeats):
        ds1 = ds.TFRecordDataset(tf_files, num_shards=num_shards, shard_id=shard_id, shard_equal_rows=True)
        ds1 = ds1.repeat(num_repeats)
        res = list()
        for data in ds1.create_dict_iterator(num_epochs=1, output_numpy=True):
            res.append(data["scalars"][0])
        return res

    worker1_res = get_res(3, 0, 2)
    worker2_res = get_res(3, 1, 2)
    worker3_res = get_res(3, 2, 2)
    # check criteria 1
    for i, _ in enumerate(worker1_res):
        assert worker1_res[i] != worker2_res[i]
        assert worker2_res[i] != worker3_res[i]
    # Confirm each worker gets same number of rows
    assert len(worker1_res) == 28
    assert len(worker1_res) == len(worker2_res)
    assert len(worker2_res) == len(worker3_res)

    worker4_res = get_res(1, 0, 1)
    assert len(worker4_res) == 40


def test_tfrecord_no_schema_columns_list():
    logger.info("test_tfrecord_no_schema_columns_list")
    data = ds.TFRecordDataset(FILES, shuffle=False, columns_list=["col_sint16"])
    row = data.create_dict_iterator(num_epochs=1, output_numpy=True).__next__()
    assert row["col_sint16"] == [-32768]

    with pytest.raises(KeyError) as info:
        _ = row["col_sint32"]
    assert "col_sint32" in str(info.value)


def test_tfrecord_schema_columns_list():
    logger.info("test_tfrecord_schema_columns_list")
    schema = ds.Schema()
    schema.add_column('col_1d', de_type=mstype.int64, shape=[2])
    schema.add_column('col_2d', de_type=mstype.int64, shape=[2, 2])
    schema.add_column('col_3d', de_type=mstype.int64, shape=[2, 2, 2])
    schema.add_column('col_binary', de_type=mstype.uint8, shape=[1])
    schema.add_column('col_float', de_type=mstype.float32, shape=[1])
    schema.add_column('col_sint16', de_type=mstype.int64, shape=[1])
    schema.add_column('col_sint32', de_type=mstype.int64, shape=[1])
    schema.add_column('col_sint64', de_type=mstype.int64, shape=[1])
    data = ds.TFRecordDataset(FILES, schema=schema, shuffle=False, columns_list=["col_sint16"])
    row = data.create_dict_iterator(num_epochs=1, output_numpy=True).__next__()
    assert row["col_sint16"] == [-32768]

    with pytest.raises(KeyError) as info:
        _ = row["col_sint32"]
    assert "col_sint32" in str(info.value)


def test_tfrecord_invalid_files():
    logger.info("test_tfrecord_invalid_files")
    valid_file = "../data/dataset/testTFTestAllTypes/test.data"
    invalid_file = "../data/dataset/testTFTestAllTypes/invalidFile.txt"
    files = [invalid_file, valid_file, SCHEMA_FILE]

    data = ds.TFRecordDataset(files, SCHEMA_FILE, shuffle=ds.Shuffle.FILES)

    with pytest.raises(RuntimeError) as info:
        _ = data.create_dict_iterator(num_epochs=1, output_numpy=True).get_next()
    assert "cannot be opened" in str(info.value)
    assert "not valid tfrecord files" in str(info.value)
    assert valid_file not in str(info.value)
    assert invalid_file in str(info.value)
    assert SCHEMA_FILE in str(info.value)

    nonexistent_file = "this/file/does/not/exist"
    files = [invalid_file, valid_file, SCHEMA_FILE, nonexistent_file]

    with pytest.raises(ValueError) as info:
        data = ds.TFRecordDataset(files, SCHEMA_FILE, shuffle=ds.Shuffle.FILES)
    assert "did not match any files" in str(info.value)
    assert valid_file not in str(info.value)
    assert invalid_file not in str(info.value)
    assert SCHEMA_FILE not in str(info.value)
    assert nonexistent_file in str(info.value)


def test_tf_wrong_schema():
    logger.info("test_tf_wrong_schema")
    files = ["../data/dataset/test_tf_file_3_images2/train-0000-of-0001.data"]
    schema = ds.Schema()
    schema.add_column('image', de_type=mstype.uint8, shape=[1])
    schema.add_column('label', de_type=mstype.int64, shape=[1])
    data1 = ds.TFRecordDataset(files, schema, shuffle=False)
    exception_occurred = False
    try:
        for _ in data1:
            pass
    except RuntimeError as e:
        exception_occurred = True
        assert "Shape in schema's column 'image' is incorrect" in str(e)

    assert exception_occurred, "test_tf_wrong_schema failed."


def test_tfrecord_invalid_columns():
    logger.info("test_tfrecord_columns_list")
    invalid_columns_list = ["not_exist"]
    data = ds.TFRecordDataset(FILES, columns_list=invalid_columns_list)
    with pytest.raises(RuntimeError) as info:
        _ = data.create_dict_iterator(num_epochs=1, output_numpy=True).__next__()
    assert "Invalid data, failed to find column name: not_exist" in str(info.value)


def test_tfrecord_exception():
    logger.info("test_tfrecord_exception")

    def exception_func(item):
        raise Exception("Error occur!")
    with pytest.raises(RuntimeError) as info:
        schema = ds.Schema()
        schema.add_column('col_1d', de_type=mstype.int64, shape=[2])
        schema.add_column('col_2d', de_type=mstype.int64, shape=[2, 2])
        schema.add_column('col_3d', de_type=mstype.int64, shape=[2, 2, 2])
        data = ds.TFRecordDataset(FILES, schema=schema, shuffle=False)
        data = data.map(operations=exception_func, input_columns=["col_1d"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
    assert "map operation: [PyFunc] failed. The corresponding data files" in str(info.value)

    with pytest.raises(RuntimeError) as info:
        schema = ds.Schema()
        schema.add_column('col_1d', de_type=mstype.int64, shape=[2])
        schema.add_column('col_2d', de_type=mstype.int64, shape=[2, 2])
        schema.add_column('col_3d', de_type=mstype.int64, shape=[2, 2, 2])
        data = ds.TFRecordDataset(FILES, schema=schema, shuffle=False)
        data = data.map(operations=exception_func, input_columns=["col_2d"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
    assert "map operation: [PyFunc] failed. The corresponding data files" in str(info.value)

    with pytest.raises(RuntimeError) as info:
        schema = ds.Schema()
        schema.add_column('col_1d', de_type=mstype.int64, shape=[2])
        schema.add_column('col_2d', de_type=mstype.int64, shape=[2, 2])
        schema.add_column('col_3d', de_type=mstype.int64, shape=[2, 2, 2])
        data = ds.TFRecordDataset(FILES, schema=schema, shuffle=False)
        data = data.map(operations=exception_func, input_columns=["col_3d"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
    assert "map operation: [PyFunc] failed. The corresponding data files" in str(info.value)


if __name__ == '__main__':
    test_tfrecord_shape()
    test_tfrecord_read_all_dataset()
    test_tfrecord_num_samples()
    test_tfrecord_num_samples2()
    test_tfrecord_shape2()
    test_tfrecord_files_basic()
    test_tfrecord_no_schema()
    test_tfrecord_pad()
    test_tfrecord_read_files()
    test_tfrecord_multi_files()
    test_tfrecord_schema()
    test_tfrecord_shuffle()
    test_tfrecord_shard()
    test_tfrecord_shard_equal_rows()
    test_tfrecord_no_schema_columns_list()
    test_tfrecord_schema_columns_list()
    test_tfrecord_invalid_files()
    test_tf_wrong_schema()
    test_tfrecord_invalid_columns()
    test_tfrecord_exception()
