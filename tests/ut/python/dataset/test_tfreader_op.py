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
import numpy as np
from util import save_and_check

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
from mindspore import log as logger
import pytest


FILES = ["../data/dataset/testTFTestAllTypes/test.data"]
DATASET_ROOT = "../data/dataset/testTFTestAllTypes/"
SCHEMA_FILE = "../data/dataset/testTFTestAllTypes/datasetSchema.json"
GENERATE_GOLDEN = False


def test_case_tf_shape():
    schema_file = "../data/dataset/testTFTestAllTypes/datasetSchemaRank0.json"
    ds1 = ds.TFRecordDataset(FILES, schema_file)
    ds1 = ds1.batch(2)
    for data in ds1.create_dict_iterator():
        logger.info(data)
    output_shape = ds1.output_shapes()
    assert (len(output_shape[-1]) == 1)


def test_case_tf_read_all_dataset():
    schema_file = "../data/dataset/testTFTestAllTypes/datasetSchemaNoRow.json"
    ds1 = ds.TFRecordDataset(FILES, schema_file)
    assert ds1.get_dataset_size() == 12
    count = 0
    for data in ds1.create_tuple_iterator():
        count += 1
    assert count == 12


def test_case_num_samples():
    schema_file = "../data/dataset/testTFTestAllTypes/datasetSchema7Rows.json"
    ds1 = ds.TFRecordDataset(FILES, schema_file, num_samples=8)
    assert ds1.get_dataset_size() == 8
    count = 0
    for data in ds1.create_dict_iterator():
        count += 1
    assert count == 8


def test_case_num_samples2():
    schema_file = "../data/dataset/testTFTestAllTypes/datasetSchema7Rows.json"
    ds1 = ds.TFRecordDataset(FILES, schema_file)
    assert ds1.get_dataset_size() == 7
    count = 0
    for data in ds1.create_dict_iterator():
        count += 1
    assert count == 7


def test_case_tf_shape_2():
    ds1 = ds.TFRecordDataset(FILES, SCHEMA_FILE)
    ds1 = ds1.batch(2)
    output_shape = ds1.output_shapes()
    assert (len(output_shape[-1]) == 2)


def test_case_tf_file():
    logger.info("reading data from: {}".format(FILES[0]))
    parameters = {"params": {}}

    data = ds.TFRecordDataset(FILES, SCHEMA_FILE, shuffle=ds.Shuffle.FILES)
    filename = "tfreader_result.npz"
    save_and_check(data, parameters, filename, generate_golden=GENERATE_GOLDEN)


def test_case_tf_file_no_schema():
    logger.info("reading data from: {}".format(FILES[0]))
    parameters = {"params": {}}

    data = ds.TFRecordDataset(FILES, shuffle=ds.Shuffle.FILES)
    filename = "tf_file_no_schema.npz"
    save_and_check(data, parameters, filename, generate_golden=GENERATE_GOLDEN)


def test_case_tf_file_pad():
    logger.info("reading data from: {}".format(FILES[0]))
    parameters = {"params": {}}

    schema_file = "../data/dataset/testTFTestAllTypes/datasetSchemaPadBytes10.json"
    data = ds.TFRecordDataset(FILES, schema_file, shuffle=ds.Shuffle.FILES)
    filename = "tf_file_padBytes10.npz"
    save_and_check(data, parameters, filename, generate_golden=GENERATE_GOLDEN)


def test_tf_files():
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


def test_tf_record_schema():
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
            assert np.array_equal(t1, t2)


def test_tf_record_shuffle():
    ds.config.set_seed(1)
    data1 = ds.TFRecordDataset(FILES, schema=SCHEMA_FILE, shuffle=ds.Shuffle.GLOBAL)
    data2 = ds.TFRecordDataset(FILES, schema=SCHEMA_FILE, shuffle=ds.Shuffle.FILES)
    data2 = data2.shuffle(10000)

    for d1, d2 in zip(data1, data2):
        for t1, t2 in zip(d1, d2):
            assert np.array_equal(t1, t2)


def skip_test_tf_record_shard():
    tf_files = ["../data/dataset/tf_file_dataset/test1.data", "../data/dataset/tf_file_dataset/test2.data",
                "../data/dataset/tf_file_dataset/test3.data", "../data/dataset/tf_file_dataset/test4.data"]

    def get_res(shard_id, num_repeats):
        data1 = ds.TFRecordDataset(tf_files, num_shards=2, shard_id=shard_id, num_samples=3,
                                   shuffle=ds.Shuffle.FILES)
        data1 = data1.repeat(num_repeats)
        res = list()
        for item in data1.create_dict_iterator():
            res.append(item["scalars"][0])
        return res

    # get separate results from two workers. the 2 results need to satisfy 2 criteria
    # 1. two workers always give different results in same epoch (e.g. wrkr1:f1&f3, wrkr2:f2&f4  in one epoch)
    # 2. with enough epochs, both workers will get the entire dataset (e,g. ep1_wrkr1: f1&f3, ep2,_wrkr1 f2&f4)
    worker1_res = get_res(0, 16)
    worker2_res = get_res(1, 16)
    # check criteria 1
    for i in range(len(worker1_res)):
        assert (worker1_res[i] != worker2_res[i])
    # check criteria 2
    assert (set(worker2_res) == set(worker1_res))
    assert (len(set(worker2_res)) == 12)


def test_tf_shard_equal_rows():
    tf_files = ["../data/dataset/tf_file_dataset/test1.data", "../data/dataset/tf_file_dataset/test2.data",
                "../data/dataset/tf_file_dataset/test3.data", "../data/dataset/tf_file_dataset/test4.data"]

    def get_res(num_shards, shard_id, num_repeats):
        ds1 = ds.TFRecordDataset(tf_files, num_shards=num_shards, shard_id=shard_id, shard_equal_rows=True)
        ds1 = ds1.repeat(num_repeats)
        res = list()
        for data in ds1.create_dict_iterator():
            res.append(data["scalars"][0])
        return res

    worker1_res = get_res(3, 0, 2)
    worker2_res = get_res(3, 1, 2)
    worker3_res = get_res(3, 2, 2)
    # check criteria 1
    for i in range(len(worker1_res)):
        assert (worker1_res[i] != worker2_res[i])
        assert (worker2_res[i] != worker3_res[i])
    assert (len(worker1_res) == 28)

    worker4_res = get_res(1, 0, 1)
    assert (len(worker4_res) == 40)


def test_case_tf_file_no_schema_columns_list():
    data = ds.TFRecordDataset(FILES, shuffle=False, columns_list=["col_sint16"])
    row = data.create_dict_iterator().get_next()
    assert row["col_sint16"] == [-32768]

    with pytest.raises(KeyError) as info:
        a = row["col_sint32"]
    assert "col_sint32" in str(info.value)


def test_tf_record_schema_columns_list():
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
    row = data.create_dict_iterator().get_next()
    assert row["col_sint16"] == [-32768]

    with pytest.raises(KeyError) as info:
        a = row["col_sint32"]
    assert "col_sint32" in str(info.value)

def test_case_invalid_files():
    valid_file = "../data/dataset/testTFTestAllTypes/test.data"
    invalid_file = "../data/dataset/testTFTestAllTypes/invalidFile.txt"
    files = [invalid_file, valid_file, SCHEMA_FILE]

    data = ds.TFRecordDataset(files, SCHEMA_FILE, shuffle=ds.Shuffle.FILES)

    with pytest.raises(RuntimeError) as info:
        row = data.create_dict_iterator().get_next()
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

if __name__ == '__main__':
    test_case_tf_shape()
    test_case_tf_read_all_dataset()
    test_case_num_samples()
    test_case_num_samples2()
    test_case_tf_shape_2()
    test_case_tf_file()
    test_case_tf_file_no_schema()
    test_case_tf_file_pad()
    test_tf_files()
    test_tf_record_schema()
    test_tf_record_shuffle()
    #test_tf_record_shard()
    test_tf_shard_equal_rows()
    test_case_tf_file_no_schema_columns_list()
    test_tf_record_schema_columns_list()
    test_case_invalid_files()
