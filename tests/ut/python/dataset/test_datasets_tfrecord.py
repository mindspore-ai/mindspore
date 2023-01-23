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
"""
Test TFRecordDataset Ops
"""
import numpy as np
import pytest

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
from mindspore import log as logger
from util import save_and_check_dict, config_get_set_seed, config_get_set_num_parallel_workers

FILES = ["../data/dataset/testTFTestAllTypes/test.data"]
DATASET_ROOT = "../data/dataset/testTFTestAllTypes/"
SCHEMA_FILE = "../data/dataset/testTFTestAllTypes/datasetSchema.json"
DATA_FILES2 = ["../data/dataset/test_tf_file_3_images2/train-0000-of-0001.data",
               "../data/dataset/test_tf_file_3_images2/train-0000-of-0002.data",
               "../data/dataset/test_tf_file_3_images2/train-0000-of-0003.data",
               "../data/dataset/test_tf_file_3_images2/train-0000-of-0004.data"]
SCHEMA_FILE2 = "../data/dataset/test_tf_file_3_images2/datasetSchema.json"
DATA_FILES3 = ["../data/dataset/tf_file_dataset/test1.data",
               "../data/dataset/tf_file_dataset/test2.data",
               "../data/dataset/tf_file_dataset/test3.data",
               "../data/dataset/tf_file_dataset/test4.data",
               "../data/dataset/tf_file_dataset/test5.data"]
SCHEMA_FILE3 = "../data/dataset/tf_file_dataset/datasetSchema.json"
SCHEMA_FILE3_NO_ROWS = "../data/dataset/tf_file_dataset/datasetSchemaNoRows.json"
SCHEMA_FILE3_ROWS_PER_SHARD = "../data/dataset/tf_file_dataset/datasetSchemaRowsPerShard.json"
GENERATE_GOLDEN = False


def test_tfrecord_shape():
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset shape
    Expectation: The dataset is processed as expected
    """
    logger.info("test_tfrecord_shape")
    schema_file = "../data/dataset/testTFTestAllTypes/datasetSchemaRank0.json"
    ds1 = ds.TFRecordDataset(FILES, schema_file)
    ds1 = ds1.batch(2)
    for data in ds1.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(data)
    output_shape = ds1.output_shapes()
    assert len(output_shape[-1]) == 1


def test_tfrecord_read_all_dataset():
    """
    Feature: TFRecordDataset
    Description: Test read all TFRecordDataset
    Expectation: The dataset is processed as expected
    """
    logger.info("test_tfrecord_read_all_dataset")
    schema_file = "../data/dataset/testTFTestAllTypes/datasetSchemaNoRow.json"
    ds1 = ds.TFRecordDataset(FILES, schema_file)
    assert ds1.get_dataset_size() == 12
    count = 0
    for _ in ds1.create_tuple_iterator(num_epochs=1):
        count += 1
    assert count == 12


def test_tfrecord_num_samples():
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset with num_samples parameter
    Expectation: The dataset is processed as expected
    """
    logger.info("test_tfrecord_num_samples")
    schema_file = "../data/dataset/testTFTestAllTypes/datasetSchema7Rows.json"
    ds1 = ds.TFRecordDataset(FILES, schema_file, num_samples=8)
    assert ds1.get_dataset_size() == 8
    count = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        count += 1
    assert count == 8


def test_tfrecord_num_samples2():
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset with no num_samples parameter
    Expectation: The dataset is processed as expected
    """
    logger.info("test_tfrecord_num_samples2")
    schema_file = "../data/dataset/testTFTestAllTypes/datasetSchema7Rows.json"
    ds1 = ds.TFRecordDataset(FILES, schema_file)
    assert ds1.get_dataset_size() == 7
    count = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        count += 1
    assert count == 7


def test_tfrecord_shape2():
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset output_shape
    Expectation: The dataset is processed as expected
    """
    logger.info("test_tfrecord_shape2")
    ds1 = ds.TFRecordDataset(FILES, SCHEMA_FILE)
    ds1 = ds1.batch(2)
    output_shape = ds1.output_shapes()
    assert len(output_shape[-1]) == 2


def test_tfrecord_files_basic():
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset files_basic
    Expectation: The dataset is processed as expected
    """
    logger.info("test_tfrecord_files_basic")

    data = ds.TFRecordDataset(FILES, SCHEMA_FILE, shuffle=ds.Shuffle.FILES)
    filename = "tfrecord_files_basic.npz"
    save_and_check_dict(data, filename, generate_golden=GENERATE_GOLDEN)


def test_tfrecord_no_schema():
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset with no schema
    Expectation: The dataset is processed as expected
    """
    logger.info("test_tfrecord_no_schema")

    data = ds.TFRecordDataset(FILES, shuffle=ds.Shuffle.FILES)
    filename = "tfrecord_no_schema.npz"
    save_and_check_dict(data, filename, generate_golden=GENERATE_GOLDEN)


def test_tfrecord_pad():
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset with pad bytes10
    Expectation: The dataset is processed as expected
    """
    logger.info("test_tfrecord_pad")

    schema_file = "../data/dataset/testTFTestAllTypes/datasetSchemaPadBytes10.json"
    data = ds.TFRecordDataset(FILES, schema_file, shuffle=ds.Shuffle.FILES)
    filename = "tfrecord_pad_bytes10.npz"
    save_and_check_dict(data, filename, generate_golden=GENERATE_GOLDEN)


def test_tfrecord_read_files():
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset read files
    Expectation: The dataset is processed as expected
    """
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
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset multi files
    Expectation: The dataset is processed as expected
    """
    logger.info("test_tfrecord_multi_files")
    data1 = ds.TFRecordDataset(DATA_FILES2, SCHEMA_FILE2, shuffle=False)
    data1 = data1.repeat(1)
    num_iter = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_iter += 1

    assert num_iter == 12


def test_tfrecord_schema():
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset schema
    Expectation: The dataset is processed as expected
    """
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
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset shuffle
    Expectation: The dataset is processed as expected
    """
    logger.info("test_tfrecord_shuffle")
    original_seed = config_get_set_seed(1)
    data1 = ds.TFRecordDataset(FILES, schema=SCHEMA_FILE, shuffle=ds.Shuffle.GLOBAL)
    data2 = ds.TFRecordDataset(FILES, schema=SCHEMA_FILE, shuffle=ds.Shuffle.FILES)
    data2 = data2.shuffle(10000)

    for d1, d2 in zip(data1, data2):
        for t1, t2 in zip(d1, d2):
            np.testing.assert_array_equal(t1.asnumpy(), t2.asnumpy())

    ds.config.set_seed(original_seed)


def test_tfrecord_shard():
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset shard
    Expectation: The dataset is processed as expected
    """
    logger.info("test_tfrecord_shard")

    def get_res(shard_id, num_repeats):
        data1 = ds.TFRecordDataset(DATA_FILES3[:-1], num_shards=2, shard_id=shard_id, num_samples=3,
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
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset shard with equal rows
    Expectation: The dataset is processed as expected
    """
    logger.info("test_tfrecord_shard_equal_rows")

    def get_res(num_shards, shard_id, num_repeats):
        ds1 = ds.TFRecordDataset(DATA_FILES3[:-1], num_shards=num_shards, shard_id=shard_id, shard_equal_rows=True)
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
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset with no schema and columns_list
    Expectation: The dataset is processed as expected
    """
    logger.info("test_tfrecord_no_schema_columns_list")
    data = ds.TFRecordDataset(FILES, shuffle=False, columns_list=["col_sint16"])
    row = data.create_dict_iterator(num_epochs=1, output_numpy=True).__next__()
    assert row["col_sint16"] == [-32768]

    with pytest.raises(KeyError) as info:
        _ = row["col_sint32"]
    assert "col_sint32" in str(info.value)


def test_tfrecord_schema_columns_list():
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset with schema and columns_list
    Expectation: The dataset is processed as expected
    """
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


def test_tfrecord_size_compression():
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset dataset size with compressed files with num_samples provided
    Expectation: The dataset is processed as expected as if not compressed
    """
    logger.info("test_tfrecord_size_compression")
    data_files_gz = []
    data_files_zlib = []

    for filename in DATA_FILES3:
        gz_filename = filename + ".gz"
        data_files_gz.append(gz_filename)

        zlib_filename = filename + ".zlib"
        data_files_zlib.append(zlib_filename)

    ds1 = ds.TFRecordDataset(DATA_FILES3, SCHEMA_FILE3,
                             num_samples=50, num_shards=5, shard_id=0, compression_type="")
    ds2 = ds.TFRecordDataset(data_files_gz, SCHEMA_FILE3,
                             num_samples=10, num_shards=5, shard_id=0, compression_type="GZIP")
    ds3 = ds.TFRecordDataset(data_files_zlib, SCHEMA_FILE3,
                             num_samples=10, num_shards=5, shard_id=0, compression_type="ZLIB")
    assert ds1.get_dataset_size() == ds2.get_dataset_size() == ds3.get_dataset_size() == 10

    ds4 = ds.TFRecordDataset(data_files_gz, SCHEMA_FILE3,
                             num_samples=16, num_shards=2, shard_id=0, compression_type="GZIP")
    ds5 = ds.TFRecordDataset(data_files_zlib, SCHEMA_FILE3,
                             num_samples=16, num_shards=2, shard_id=0, compression_type="ZLIB")
    assert ds4.get_dataset_size() == ds5.get_dataset_size() == 16


def test_tfrecord_basic_compression():
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset with compressed files (GZIP and ZLIB)
    Expectation: The dataset is processed as expected as if not compressed
    """
    logger.info("test_tfrecord_basic_compression")
    data_files_gz = []
    data_files_zlib = []

    for filename in DATA_FILES3:
        gz_filename = filename + ".gz"
        data_files_gz.append(gz_filename)

        zlib_filename = filename + ".zlib"
        data_files_zlib.append(zlib_filename)

    data1 = ds.TFRecordDataset(data_files_gz, SCHEMA_FILE3, num_samples=50, shuffle=False, compression_type='GZIP')
    data2 = ds.TFRecordDataset(data_files_zlib, SCHEMA_FILE3, num_samples=50, shuffle=False, compression_type='ZLIB')
    data3 = ds.TFRecordDataset(DATA_FILES3, SCHEMA_FILE3, num_samples=50, shuffle=False)

    num_iter = 0

    for item1, item2, item3 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                                   data2.create_dict_iterator(num_epochs=1, output_numpy=True),
                                   data3.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(item1['scalars'], item2['scalars'])
        np.testing.assert_array_equal(item2['scalars'], item3['scalars'])
        num_iter += 1

    assert num_iter == 50


def test_tfrecord_compression_with_other_ops():
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset with compressed files (GZIP and ZLIB) with batch op
    Expectation: The dataset is processed as expected as if not compressed
    """
    logger.info("test_tfrecord_basic_compression")
    data_files_gz = []
    data_files_zlib = []
    batch_size = 3
    drop_remainder = True
    num_workers = 4
    original_num_workers = config_get_set_num_parallel_workers(num_workers)

    for filename in DATA_FILES3:
        gz_filename = filename + ".gz"
        data_files_gz.append(gz_filename)

        zlib_filename = filename + ".zlib"
        data_files_zlib.append(zlib_filename)

    data1 = ds.TFRecordDataset(data_files_gz, SCHEMA_FILE3, num_samples=50,
                               shuffle=False, compression_type='GZIP')
    data2 = ds.TFRecordDataset(data_files_zlib, SCHEMA_FILE3, num_samples=50,
                               shuffle=False, compression_type='ZLIB')
    data3 = ds.TFRecordDataset(
        DATA_FILES3, SCHEMA_FILE3, num_samples=50, shuffle=False)
    data1 = data1.batch(num_parallel_workers=num_workers, batch_size=batch_size, drop_remainder=drop_remainder)
    data2 = data2.batch(num_parallel_workers=num_workers, batch_size=batch_size, drop_remainder=drop_remainder)
    data3 = data3.batch(num_parallel_workers=num_workers, batch_size=batch_size, drop_remainder=drop_remainder)

    num_iter = 0

    for item1, item2, item3 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                                   data2.create_dict_iterator(num_epochs=1, output_numpy=True),
                                   data3.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(item1['scalars'], item2['scalars'])
        np.testing.assert_array_equal(item2['scalars'], item3['scalars'])
        num_iter += 1

    assert num_iter == 16
    ds.config.set_num_parallel_workers(original_num_workers)


def test_tfrecord_compression_no_schema():
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset with compressed files (GZIP and ZLIB) with no schema
    Expectation: The dataset is processed as expected as if not compressed
    """
    logger.info("test_tfrecord_compression_no_schemas")
    data_files_gz = []
    data_files_zlib = []

    for filename in DATA_FILES3:
        gz_filename = filename + ".gz"
        data_files_gz.append(gz_filename)

        zlib_filename = filename + ".zlib"
        data_files_zlib.append(zlib_filename)

    data1 = ds.TFRecordDataset(data_files_gz, num_samples=50,
                               shuffle=False, compression_type='GZIP')
    data2 = ds.TFRecordDataset(data_files_zlib, num_samples=50,
                               shuffle=False, compression_type='ZLIB')
    data3 = ds.TFRecordDataset(DATA_FILES3, num_samples=50, shuffle=False)
    num_iter = 0

    for item1, item2, item3 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                                   data2.create_dict_iterator(num_epochs=1, output_numpy=True),
                                   data3.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(item1['scalars'], item2['scalars'])
        np.testing.assert_array_equal(item2['scalars'], item3['scalars'])
        num_iter += 1

    assert num_iter == 50


def test_tfrecord_compression_shard_exact():
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset with compressed files (GZIP and ZLIB) with num_shards == num files
    Expectation: The dataset is processed as expected as if not compressed
    """
    logger.info("test_tfrecord_compression_shard_exact")
    data_files_gz = []
    data_files_zlib = []

    original_num_workers = config_get_set_num_parallel_workers(2)

    for filename in DATA_FILES3:
        gz_filename = filename + ".gz"
        data_files_gz.append(gz_filename)

        zlib_filename = filename + ".zlib"
        data_files_zlib.append(zlib_filename)

    def get_res(shard_id, num_repeats, shuffle, compression_type):
        if compression_type == 'GZIP':
            data1 = ds.TFRecordDataset(data_files_gz, num_shards=5, shard_id=shard_id, num_samples=10,
                                       shuffle=shuffle, compression_type=compression_type)
        elif compression_type == 'ZLIB':
            data1 = ds.TFRecordDataset(data_files_zlib, num_shards=5, shard_id=shard_id, num_samples=10,
                                       shuffle=shuffle, compression_type=compression_type)
        else:
            data1 = ds.TFRecordDataset(DATA_FILES3, num_shards=5, shard_id=shard_id, num_samples=10,
                                       shuffle=shuffle, compression_type=compression_type)
        data1 = data1.repeat(num_repeats)
        res = list()
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            res.append(item['scalars'][0])
        return res

    worker_uncomp_false = get_res(0, 5, False, '')
    worker_gzip_false = get_res(0, 5, False, 'GZIP')
    worker_zlib_false = get_res(0, 5, False, 'ZLIB')
    # Confirm each worker gets 10x5=50 rows
    assert len(worker_uncomp_false) == len(worker_gzip_false) == len(worker_zlib_false) == 50
    assert worker_uncomp_false == worker_gzip_false == worker_zlib_false == list(range(1, 11)) * 5

    ds.config.set_num_parallel_workers(original_num_workers)


def test_tfrecord_compression_shard_odd():
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset with compressed files (ZLIB) with num files % num_shards != 0
    Expectation: The dataset is processed as expected as if not compressed
    """
    logger.info("test_tfrecord_compression_shard_odd")
    data_files_zlib = []

    original_seed = config_get_set_seed(1)
    original_num_workers = config_get_set_num_parallel_workers(4)

    for filename in DATA_FILES3:
        zlib_filename = filename + ".zlib"
        data_files_zlib.append(zlib_filename)

    def get_res(shard_id, num_repeats):
        data1 = ds.TFRecordDataset(data_files_zlib, SCHEMA_FILE3, num_shards=3, shard_id=shard_id,
                                   num_samples=10, shuffle=ds.Shuffle.FILES,
                                   compression_type='ZLIB')
        data1 = data1.repeat(num_repeats)
        res = list()
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            res.append(item['scalars'][0])
        return res

    worker1_res = get_res(0, 20)
    worker2_res = get_res(1, 20)
    worker3_res = get_res(2, 20)

    # Confirm each worker gets 10x20=200 rows
    assert len(worker1_res) == len(worker2_res) == len(worker3_res) == 200
    # check criteria 1
    for i, _ in enumerate(worker1_res):
        assert worker1_res[i] != worker2_res[i]
        assert worker1_res[i] != worker3_res[i]
        assert worker2_res[i] != worker3_res[i]
    # check criteria 2
    assert set(worker1_res) == set(worker2_res) == set(worker3_res) == set(range(1, 51))

    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_workers)


def test_tfrecord_compression_shard_even():
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset with compressed files (GZIP) with num files % num_shards == 0
    Expectation: The dataset is processed as expected as if not compressed
    """
    logger.info("test_tfrecord_compression_shard_even")
    data_files_gz = []

    original_seed = config_get_set_seed(1)
    original_num_workers = config_get_set_num_parallel_workers(5)

    for filename in DATA_FILES3[:-1]:
        gz_filename = filename + ".gz"
        data_files_gz.append(gz_filename)

    def get_res(shard_id, num_repeats):
        data1 = ds.TFRecordDataset(data_files_gz, SCHEMA_FILE3, num_shards=2, shard_id=shard_id,
                                   num_samples=20, shuffle=ds.Shuffle.GLOBAL,
                                   compression_type='GZIP')
        data1 = data1.repeat(num_repeats)
        res = list()
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            res.append(item['scalars'][0])
        return res

    worker1_res = get_res(0, 16)
    worker2_res = get_res(1, 16)

    # Confirm each worker gets 20x16=320 rows
    assert len(worker1_res) == len(worker2_res) == 320
    # check criteria 1
    for i, _ in enumerate(worker1_res):
        assert worker1_res[i] != worker2_res[i]
    # check criteria 2
    assert set(worker1_res) == set(worker2_res) == set(range(1, 41))

    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_workers)


def test_tfrecord_compression_shape_and_size_no_num_samples():
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset shape and dataset size with compressed files and no num_samples provided
    Expectation: The dataset is processed as expected as if not compressed
    """
    logger.info("test_tfrecord_compression_shape_and_size_no_num_samples")
    data_files_gz = []
    data_files_zlib = []

    for filename in DATA_FILES3:
        gz_filename = filename + ".gz"
        data_files_gz.append(gz_filename)

        zlib_filename = filename + ".zlib"
        data_files_zlib.append(zlib_filename)

    ds1 = ds.TFRecordDataset(DATA_FILES3, compression_type="")
    ds2 = ds.TFRecordDataset(data_files_gz, compression_type="GZIP")
    ds3 = ds.TFRecordDataset(data_files_zlib, compression_type="ZLIB")
    assert ds1.get_dataset_size() == ds2.get_dataset_size() == ds3.get_dataset_size() == 50

    ds1 = ds1.batch(3)
    ds2 = ds2.batch(3)
    ds3 = ds3.batch(3)
    count1 = count2 = count3 = 0
    for _ in ds1.create_dict_iterator(num_epochs=1, output_numpy=True):
        count1 += 1
    for _ in ds2.create_dict_iterator(num_epochs=1, output_numpy=True):
        count2 += 1
    for _ in ds3.create_dict_iterator(num_epochs=1, output_numpy=True):
        count3 += 1
    assert ds1.output_shapes() == ds2.output_shapes() == ds3.output_shapes()
    assert count1 == count2 == count3 == 17


def test_tfrecord_compression_shape_and_size_no_num_samples_with_schema_numrows():
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset shape and dataset size with compressed files, no num_samples
    but numRows schema provided
    Expectation: The dataset is processed as expected as if not compressed
    """
    logger.info("test_tfrecord_compression_shape_and_size_no_num_samples_with_schema_numrows")
    data_files_gz = []
    data_files_zlib = []

    for filename in DATA_FILES3:
        gz_filename = filename + ".gz"
        data_files_gz.append(gz_filename)

        zlib_filename = filename + ".zlib"
        data_files_zlib.append(zlib_filename)

    ds1 = ds.TFRecordDataset(DATA_FILES3, SCHEMA_FILE3, compression_type="")
    ds2 = ds.TFRecordDataset(data_files_gz, SCHEMA_FILE3, compression_type="GZIP")
    ds3 = ds.TFRecordDataset(data_files_zlib, SCHEMA_FILE3, compression_type="ZLIB")
    assert ds1.get_dataset_size() == ds2.get_dataset_size() == ds3.get_dataset_size() == 50

    ds1 = ds1.batch(3)
    ds2 = ds2.batch(3)
    ds3 = ds3.batch(3)
    count1 = count2 = count3 = 0
    for _ in ds1.create_dict_iterator(num_epochs=1, output_numpy=True):
        count1 += 1
    for _ in ds2.create_dict_iterator(num_epochs=1, output_numpy=True):
        count2 += 1
    for _ in ds3.create_dict_iterator(num_epochs=1, output_numpy=True):
        count3 += 1
    assert ds1.output_shapes() == ds2.output_shapes() == ds3.output_shapes()
    assert count1 == count2 == count3 == 17


def test_tfrecord_compression_shape_and_size_no_num_samples_with_schema_nonumrows():
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset shape and dataset size with compressed files, no num_samples,
    and no numRows schema provided
    Expectation: The dataset is processed as expected as if not compressed
    """
    logger.info("test_tfrecord_compression_shape_and_size_no_num_samples_with_schema_nonumrows")
    data_files_gz = []
    data_files_zlib = []

    for filename in DATA_FILES3:
        gz_filename = filename + ".gz"
        data_files_gz.append(gz_filename)

        zlib_filename = filename + ".zlib"
        data_files_zlib.append(zlib_filename)

    ds1 = ds.TFRecordDataset(DATA_FILES3, SCHEMA_FILE3_NO_ROWS, compression_type="")
    ds2 = ds.TFRecordDataset(data_files_gz, SCHEMA_FILE3_NO_ROWS, compression_type="GZIP")
    ds3 = ds.TFRecordDataset(data_files_zlib, SCHEMA_FILE3_NO_ROWS, compression_type="ZLIB")
    assert ds1.get_dataset_size() == ds2.get_dataset_size() == ds3.get_dataset_size() == 50

    ds1 = ds1.batch(3)
    ds2 = ds2.batch(3)
    ds3 = ds3.batch(3)
    count1 = count2 = count3 = 0
    for _ in ds1.create_dict_iterator(num_epochs=1, output_numpy=True):
        count1 += 1
    for _ in ds2.create_dict_iterator(num_epochs=1, output_numpy=True):
        count2 += 1
    for _ in ds3.create_dict_iterator(num_epochs=1, output_numpy=True):
        count3 += 1
    assert ds1.output_shapes() == ds2.output_shapes() == ds3.output_shapes()
    assert count1 == count2 == count3 == 17


def test_tfrecord_compression_no_num_samples_with_schema_numrows_shard():
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset with compressed files (GZIP and ZLIB) with numRows in schema provided
    Expectation: The dataset is processed as expected as if not compressed
    """
    logger.info("test_tfrecord_compression_no_num_samples_with_schema_numrows_shard")
    data_files_gz = []
    data_files_zlib = []

    original_num_workers = config_get_set_num_parallel_workers(2)

    for filename in DATA_FILES3:
        gz_filename = filename + ".gz"
        data_files_gz.append(gz_filename)

        zlib_filename = filename + ".zlib"
        data_files_zlib.append(zlib_filename)

    def get_res(shard_id, num_repeats, shuffle, compression_type):
        if compression_type == 'GZIP':
            data1 = ds.TFRecordDataset(data_files_gz, SCHEMA_FILE3_ROWS_PER_SHARD, num_shards=5, shard_id=shard_id,
                                       shuffle=shuffle, compression_type=compression_type)
        elif compression_type == 'ZLIB':
            data1 = ds.TFRecordDataset(data_files_zlib, SCHEMA_FILE3_ROWS_PER_SHARD, num_shards=5, shard_id=shard_id,
                                       shuffle=shuffle, compression_type=compression_type)
        else:
            data1 = ds.TFRecordDataset(DATA_FILES3, SCHEMA_FILE3_ROWS_PER_SHARD, num_shards=5, shard_id=shard_id,
                                       shuffle=shuffle, compression_type=compression_type)
        data1 = data1.repeat(num_repeats)
        res = list()
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            res.append(item['scalars'][0])
        return res

    worker_uncomp_false = get_res(0, 5, False, '')
    worker_gzip_false = get_res(0, 5, False, 'GZIP')
    worker_zlib_false = get_res(0, 5, False, 'ZLIB')
    # Confirm each worker gets 10x5=50 rows
    assert len(worker_uncomp_false) == len(worker_gzip_false) == len(worker_zlib_false) == 50
    assert worker_uncomp_false == worker_gzip_false == worker_zlib_false == list(range(1, 11)) * 5

    ds.config.set_num_parallel_workers(original_num_workers)


def test_tfrecord_compression_shard_no_num_samples():
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset shard with compressed files and no num_samples
    Expectation: The dataset is processed as expected as if not compressed
    """
    logger.info("test_tfrecord_compression_shard_no_num_samples")

    original_seed = config_get_set_seed(1)
    original_num_workers = config_get_set_num_parallel_workers(3)

    data_files_gz = []
    data_files_zlib = []

    for filename in DATA_FILES3:
        gz_filename = filename + ".gz"
        data_files_gz.append(gz_filename)

        zlib_filename = filename + ".zlib"
        data_files_zlib.append(zlib_filename)

    def get_res(shard_id, num_repeats, compression_type):
        if compression_type == 'GZIP':
            data1 = ds.TFRecordDataset(data_files_gz, num_shards=4, shard_id=shard_id,
                                       shuffle=ds.Shuffle.FILES, compression_type=compression_type)
        elif compression_type == 'ZLIB':
            data1 = ds.TFRecordDataset(data_files_zlib, num_shards=4, shard_id=shard_id,
                                       shuffle=ds.Shuffle.FILES, compression_type=compression_type)
        else:
            data1 = ds.TFRecordDataset(DATA_FILES3, num_shards=4, shard_id=shard_id,
                                       shuffle=ds.Shuffle.FILES, compression_type=compression_type)
        data1 = data1.repeat(num_repeats)
        res = list()
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            res.append(item['scalars'][0])
        return res

    worker_uncomp_files = get_res(0, 20, '')
    worker1_gzip_files = get_res(0, 20, 'GZIP')
    worker_zlib_files = get_res(0, 20, 'ZLIB')
    # Confirm each worker gets 20x20 = 400 rows
    assert len(worker_uncomp_files) == len(worker1_gzip_files) == len(worker_zlib_files) == 400
    assert worker_uncomp_files == worker1_gzip_files == worker_zlib_files
    assert set(worker1_gzip_files) == set(range(1, 51))

    worker2_gzip_files = get_res(1, 20, 'GZIP')
    worker3_gzip_files = get_res(2, 20, 'GZIP')
    worker4_gzip_files = get_res(3, 20, 'GZIP')
    assert len(worker2_gzip_files) == len(worker3_gzip_files) == len(worker4_gzip_files) == 200

    for i, _ in enumerate(worker2_gzip_files):
        assert worker2_gzip_files[i] != worker3_gzip_files[i]
        assert worker2_gzip_files[i] != worker4_gzip_files[i]
        assert worker3_gzip_files[i] != worker4_gzip_files[i]

    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_workers)


def test_tfrecord_compression_shard_equal_rows_no_num_samples():
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset shard with equal rows with compressed files and no num_samples
    Expectation: The dataset is processed as expected as if not compressed
    """
    logger.info("test_tfrecord_compression_shard_equal_rows_no_num_samples")

    original_seed = config_get_set_seed(1)
    original_num_workers = config_get_set_num_parallel_workers(4)

    data_files_gz = []
    data_files_zlib = []

    for filename in DATA_FILES3:
        gz_filename = filename + ".gz"
        data_files_gz.append(gz_filename)

        zlib_filename = filename + ".zlib"
        data_files_zlib.append(zlib_filename)

    def get_res(shard_id, num_repeats, compression_type):
        if compression_type == 'GZIP':
            data1 = ds.TFRecordDataset(data_files_gz, num_shards=3, shard_id=shard_id, shard_equal_rows=True,
                                       shuffle=ds.Shuffle.GLOBAL, compression_type=compression_type)
        elif compression_type == 'ZLIB':
            data1 = ds.TFRecordDataset(data_files_zlib, num_shards=3, shard_id=shard_id, shard_equal_rows=True,
                                       shuffle=ds.Shuffle.GLOBAL, compression_type=compression_type)
        else:
            data1 = ds.TFRecordDataset(DATA_FILES3, num_shards=3, shard_id=shard_id, shard_equal_rows=True,
                                       shuffle=ds.Shuffle.GLOBAL, compression_type=compression_type)
        data1 = data1.repeat(num_repeats)
        res = list()
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            res.append(item['scalars'][0])
        return res

    worker_uncomp_global = get_res(0, 20, '')
    worker_gzip_global = get_res(0, 20, 'GZIP')
    worker1_zlib_global = get_res(0, 20, 'ZLIB')
    # Confirm each worker gets 17x20 = 340 rows
    assert len(worker_uncomp_global) == len(worker_gzip_global) == len(worker1_zlib_global) == 340
    assert worker_uncomp_global == worker_gzip_global == worker1_zlib_global
    # Confirm for one worker that after 20 repeats, all 50 unique rows are included in the dataset pipeline
    assert set(worker1_zlib_global) == set(range(1, 51))

    worker2_zlib_global = get_res(1, 20, 'ZLIB')
    worker3_zlib_global = get_res(2, 20, 'ZLIB')
    assert len(worker1_zlib_global) == len(worker2_zlib_global) == len(worker3_zlib_global) == 340

    for i, _ in enumerate(worker1_zlib_global):
        assert worker1_zlib_global[i] != worker2_zlib_global[i]
        assert worker1_zlib_global[i] != worker3_zlib_global[i]
        assert worker2_zlib_global[i] != worker3_zlib_global[i]

    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_workers)


def test_tfrecord_compression_invalid_inputs():
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset with compressed files (GZIP), but invalid inputs
    Expectation: Error is raised as expected
    """
    logger.info("test_tfrecord_compression_invalid_inputs")
    data_files_gz = []

    for filename in DATA_FILES3:
        gz_filename = filename + ".gz"
        data_files_gz.append(gz_filename)

    # must meet minimum sample requirement
    data1 = ds.TFRecordDataset(data_files_gz, num_shards=3, shard_id=0,
                               num_samples=40, shuffle=False, compression_type='GZIP')
    data1 = data1.repeat(1)
    with pytest.raises(RuntimeError) as info:
        for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass
    assert "does not meet minimum rows per shard requirement" in str(info.value)

    # must meet minimum sample requirement (using schema)
    data2 = ds.TFRecordDataset(data_files_gz, SCHEMA_FILE3, num_shards=3, shard_id=0,
                               shuffle=False, compression_type='GZIP')
    data2 = data2.repeat(1)
    with pytest.raises(RuntimeError) as info:
        for _ in data2.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass
    assert "does not meet minimum rows per shard requirement" in str(info.value)

    # number of dataset files cannot be less than num_shards
    with pytest.raises(ValueError) as info:
        _ = ds.TFRecordDataset(data_files_gz, SCHEMA_FILE3, num_shards=6, shard_id=0,
                               num_samples=10, shuffle=False, compression_type='GZIP')
    assert "When compression_type is provided, the number of dataset files cannot be less than num_shards" in str(
        info.value)

    # invalid num_samples
    with pytest.raises(ValueError) as info:
        _ = ds.TFRecordDataset(DATA_FILES3[0] + ".gz", SCHEMA_FILE3, num_samples=-1, compression_type="GZIP")
    assert "num_samples exceeds the boundary" in str(info.value)

    # check compression_type
    with pytest.raises(ValueError) as info:
        _ = ds.TFRecordDataset(data_files_gz, SCHEMA_FILE3, num_shards=3, shard_id=0,
                               shard_equal_rows=True, shuffle=False, compression_type='ZIP')
    assert "Input compression_type can only be either '' (no compression), 'ZLIB', or 'GZIP'" in str(info.value)


def test_tfrecord_invalid_files():
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset with invalid files
    Expectation: Correct error is thrown as expected
    """
    logger.info("test_tfrecord_invalid_files")
    valid_file = "../data/dataset/testTFTestAllTypes/test.data"
    invalid_file = "../data/dataset/testTFTestAllTypes/invalidFile.txt"
    files = [invalid_file, valid_file, SCHEMA_FILE]

    data = ds.TFRecordDataset(files, SCHEMA_FILE, shuffle=ds.Shuffle.FILES)

    with pytest.raises(RuntimeError) as info:
        _ = data.create_dict_iterator(num_epochs=1, output_numpy=True).__next__()
    assert "cannot be opened" in str(info.value)
    assert "not valid TFRecordDataset files" in str(info.value)
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
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset with wrong schema
    Expectation: Error is raised as expected
    """
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
        assert "Data dimensions of 'image' do not match" in str(e)

    assert exception_occurred, "test_tf_wrong_schema failed."


def test_tfrecord_invalid_columns():
    """
    Feature: TFRecordDataset
    Description: Test TFRecordDataset with invalid columns
    Expectation: Error is raised as expected
    """
    logger.info("test_tfrecord_invalid_columns")
    invalid_columns_list = ["not_exist"]
    data = ds.TFRecordDataset(FILES, columns_list=invalid_columns_list)
    with pytest.raises(RuntimeError) as info:
        _ = data.create_dict_iterator(num_epochs=1, output_numpy=True).__next__()
    assert "Invalid columns_list, tfrecord file failed to find column name: not_exist" in str(info.value)


def test_tfrecord_exception():
    """
    Feature: TFRecordDataset
    Description: Test error cases for TFRecordDataset
    Expectation: Correct error is thrown as expected
    """
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
    assert "map operation: [PyFunc] failed. The corresponding data file is" in str(info.value)

    with pytest.raises(RuntimeError) as info:
        schema = ds.Schema()
        schema.add_column('col_1d', de_type=mstype.int64, shape=[2])
        schema.add_column('col_2d', de_type=mstype.int64, shape=[2, 2])
        schema.add_column('col_3d', de_type=mstype.int64, shape=[2, 2, 2])
        data = ds.TFRecordDataset(FILES, schema=schema, shuffle=False)
        data = data.map(operations=exception_func, input_columns=["col_2d"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
    assert "map operation: [PyFunc] failed. The corresponding data file is" in str(info.value)

    with pytest.raises(RuntimeError) as info:
        schema = ds.Schema()
        schema.add_column('col_1d', de_type=mstype.int64, shape=[2])
        schema.add_column('col_2d', de_type=mstype.int64, shape=[2, 2])
        schema.add_column('col_3d', de_type=mstype.int64, shape=[2, 2, 2])
        data = ds.TFRecordDataset(FILES, schema=schema, shuffle=False)
        data = data.map(operations=exception_func, input_columns=["col_3d"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
    assert "map operation: [PyFunc] failed. The corresponding data file is" in str(info.value)


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
    test_tfrecord_size_compression()
    test_tfrecord_basic_compression()
    test_tfrecord_compression_with_other_ops()
    test_tfrecord_compression_no_schema()
    test_tfrecord_compression_shard_exact()
    test_tfrecord_compression_shard_odd()
    test_tfrecord_compression_shard_even()
    test_tfrecord_compression_shape_and_size_no_num_samples()
    test_tfrecord_compression_shape_and_size_no_num_samples_with_schema_numrows()
    test_tfrecord_compression_shape_and_size_no_num_samples_with_schema_nonumrows()
    test_tfrecord_compression_no_num_samples_with_schema_numrows_shard()
    test_tfrecord_compression_shard_no_num_samples()
    test_tfrecord_compression_shard_equal_rows_no_num_samples()
    test_tfrecord_compression_invalid_inputs()
    test_tfrecord_invalid_files()
    test_tf_wrong_schema()
    test_tfrecord_invalid_columns()
    test_tfrecord_exception()
