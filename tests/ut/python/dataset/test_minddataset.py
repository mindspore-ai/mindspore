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
This is the test module for mindrecord
"""
import collections
import json
import math
import os
import re
import string
import pytest
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import log as logger
from mindspore.dataset.vision import Inter
from mindspore.mindrecord import FileWriter
from util import config_get_set_seed

FILES_NUM = 4
CV_DIR_NAME = "../data/mindrecord/testImageNetData"
NLP_FILE_POS = "../data/mindrecord/testAclImdbData/pos"
NLP_FILE_VOCAB = "../data/mindrecord/testAclImdbData/vocab.txt"

@pytest.fixture
def add_and_remove_cv_file():
    """add/remove cv file"""
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    paths = ["{}{}".format(file_name, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    try:
        for x in paths:
            if os.path.exists("{}".format(x)):
                os.remove("{}".format(x))
            if os.path.exists("{}.db".format(x)):
                os.remove("{}.db".format(x))
        writer = FileWriter(file_name, FILES_NUM)
        data = get_data(CV_DIR_NAME)
        cv_schema_json = {"id": {"type": "int32"},
                          "file_name": {"type": "string"},
                          "label": {"type": "int32"},
                          "data": {"type": "bytes"}}
        writer.add_schema(cv_schema_json, "img_schema")
        writer.add_index(["file_name", "label"])
        writer.write_raw_data(data)
        writer.commit()
        yield "yield_cv_data"
    except Exception as error:
        for x in paths:
            os.remove("{}".format(x))
            os.remove("{}.db".format(x))
        raise error
    else:
        for x in paths:
            os.remove("{}".format(x))
            os.remove("{}.db".format(x))


@pytest.fixture
def add_and_remove_nlp_file():
    """add/remove nlp file"""
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    paths = ["{}{}".format(file_name, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    try:
        for x in paths:
            if os.path.exists("{}".format(x)):
                os.remove("{}".format(x))
            if os.path.exists("{}.db".format(x)):
                os.remove("{}.db".format(x))
        writer = FileWriter(file_name, FILES_NUM)
        data = [x for x in get_nlp_data(NLP_FILE_POS, NLP_FILE_VOCAB, 10)]
        nlp_schema_json = {"id": {"type": "string"}, "label": {"type": "int32"},
                           "rating": {"type": "float32"},
                           "input_ids": {"type": "int64",
                                         "shape": [-1]},
                           "input_mask": {"type": "int64",
                                          "shape": [1, -1]},
                           "segment_ids": {"type": "int64",
                                           "shape": [2, -1]}
                           }
        writer.set_header_size(1 << 14)
        writer.set_page_size(1 << 15)
        writer.add_schema(nlp_schema_json, "nlp_schema")
        writer.add_index(["id", "rating"])
        writer.write_raw_data(data)
        writer.commit()
        yield "yield_nlp_data"
    except Exception as error:
        for x in paths:
            os.remove("{}".format(x))
            os.remove("{}.db".format(x))
        raise error
    else:
        for x in paths:
            os.remove("{}".format(x))
            os.remove("{}.db".format(x))


@pytest.fixture
def add_and_remove_three_nlp_file_with_sample_10_0_5():
    """add/remove nlp file"""
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    paths = [file_name + "_with_sample_10",
             file_name + "_with_sample_0",
             file_name + "_with_sample_5"]
    try:
        def create_mindrecord_file(file_name, num_samples):
            if os.path.exists(file_name):
                os.remove("{}".format(file_name))
            writer = FileWriter(file_name)
            if num_samples > 0:
                data = [x for x in get_nlp_data(NLP_FILE_POS, NLP_FILE_VOCAB, num_samples)]
            nlp_schema_json = {"id": {"type": "string"}, "label": {"type": "int32"},
                               "rating": {"type": "float32"},
                               "input_ids": {"type": "int64",
                                             "shape": [-1]},
                               "input_mask": {"type": "int64",
                                              "shape": [1, -1]},
                               "segment_ids": {"type": "int64",
                                               "shape": [2, -1]}
                               }
            writer.set_header_size(1 << 14)
            writer.set_page_size(1 << 15)
            writer.add_schema(nlp_schema_json, "nlp_schema")
            writer.add_index(["id", "rating"])
            if num_samples > 0:
                writer.write_raw_data(data)
            writer.commit()

        create_mindrecord_file(paths[0], 10)
        create_mindrecord_file(paths[1], 0)
        create_mindrecord_file(paths[2], 5)

        yield "yield_nlp_data"
    except Exception as error:
        for x in paths:
            os.remove("{}".format(x))
            os.remove("{}.db".format(x))
        raise error
    else:
        for x in paths:
            os.remove("{}".format(x))
            os.remove("{}.db".format(x))


@pytest.fixture
def add_and_remove_nlp_compress_file():
    """add/remove nlp file"""
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    paths = ["{}{}".format(file_name, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    try:
        for x in paths:
            if os.path.exists("{}".format(x)):
                os.remove("{}".format(x))
            if os.path.exists("{}.db".format(x)):
                os.remove("{}.db".format(x))
        writer = FileWriter(file_name, FILES_NUM)
        data = []
        for row_id in range(16):
            data.append({
                "label": row_id,
                "array_a": np.reshape(np.array([0, 1, -1, 127, -128, 128, -129,
                                                255, 256, -32768, 32767, -32769, 32768, -2147483648,
                                                2147483647], dtype=np.int32), [-1]),
                "array_b": np.reshape(np.array([0, 1, -1, 127, -128, 128, -129, 255,
                                                256, -32768, 32767, -32769, 32768,
                                                -2147483648, 2147483647, -2147483649, 2147483649,
                                                -922337036854775808, 9223372036854775807]), [1, -1]),
                "array_c": str.encode("nlp data"),
                "array_d": np.reshape(np.array([[-10, -127], [10, 127]]), [2, -1])
            })
        nlp_schema_json = {"label": {"type": "int32"},
                           "array_a": {"type": "int32",
                                       "shape": [-1]},
                           "array_b": {"type": "int64",
                                       "shape": [1, -1]},
                           "array_c": {"type": "bytes"},
                           "array_d": {"type": "int64",
                                       "shape": [2, -1]}
                           }
        writer.set_header_size(1 << 14)
        writer.set_page_size(1 << 15)
        writer.add_schema(nlp_schema_json, "nlp_schema")
        writer.write_raw_data(data)
        writer.commit()
        yield "yield_nlp_data"
    except Exception as error:
        for x in paths:
            os.remove("{}".format(x))
            os.remove("{}.db".format(x))
        raise error
    else:
        for x in paths:
            os.remove("{}".format(x))
            os.remove("{}.db".format(x))


def test_nlp_compress_data(add_and_remove_nlp_compress_file):
    """
    Feature: MindDataset
    Description: Test compressing NLP MindDataset
    Expectation: Output is equal to the expected output
    """
    data = []
    for row_id in range(16):
        data.append({
            "label": row_id,
            "array_a": np.reshape(np.array([0, 1, -1, 127, -128, 128, -129,
                                            255, 256, -32768, 32767, -32769, 32768, -2147483648,
                                            2147483647], dtype=np.int32), [-1]),
            "array_b": np.reshape(np.array([0, 1, -1, 127, -128, 128, -129, 255,
                                            256, -32768, 32767, -32769, 32768,
                                            -2147483648, 2147483647, -2147483649, 2147483649,
                                            -922337036854775808, 9223372036854775807]), [1, -1]),
            "array_c": str.encode("nlp data"),
            "array_d": np.reshape(np.array([[-10, -127], [10, 127]]), [2, -1])
        })
    num_readers = 1
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset(
        file_name + "0", None, num_readers, shuffle=False)
    assert data_set.get_dataset_size() == 16
    num_iter = 0
    for x, item in zip(data, data_set.create_dict_iterator(num_epochs=1, output_numpy=True)):
        assert (item["array_a"] == x["array_a"]).all()
        assert (item["array_b"] == x["array_b"]).all()
        assert item["array_c"].tobytes() == x["array_c"]
        assert (item["array_d"] == x["array_d"]).all()
        assert item["label"] == x["label"]
        num_iter += 1
    assert num_iter == 16


def test_cv_minddataset_writer_tutorial():
    """
    Feature: MindDataset
    Description: Test MindDataset FileWriter basic usage
    Expectation: Runs successfully
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    paths = ["{}{}".format(file_name, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    try:
        for x in paths:
            if os.path.exists("{}".format(x)):
                os.remove("{}".format(x))
            if os.path.exists("{}.db".format(x)):
                os.remove("{}.db".format(x))
        writer = FileWriter(file_name, FILES_NUM)
        data = get_data(CV_DIR_NAME)
        cv_schema_json = {"file_name": {"type": "string"}, "label": {"type": "int32"},
                          "data": {"type": "bytes"}}
        writer.add_schema(cv_schema_json, "img_schema")
        writer.add_index(["file_name", "label"])
        writer.write_raw_data(data)
        writer.commit()
    except Exception as error:
        for x in paths:
            os.remove("{}".format(x))
            os.remove("{}.db".format(x))
        raise error
    else:
        for x in paths:
            os.remove("{}".format(x))
            os.remove("{}.db".format(x))


def test_cv_minddataset_partition_tutorial(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test partition (using num_shards and shard_id) on MindDataset basic usage
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    def partitions(num_shards):
        for partition_id in range(num_shards):
            data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                                      num_shards=num_shards, shard_id=partition_id)
            num_iter = 0
            for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
                logger.info("-------------- partition : {} ------------------------".format(partition_id))
                logger.info("-------------- item[file_name]: {}-----------------------".format(item["file_name"]))
                logger.info("-------------- item[label]: {} -----------------------".format(item["label"]))
                num_iter += 1
        return num_iter

    assert partitions(4) == 3
    assert partitions(5) == 2
    assert partitions(9) == 2


def test_cv_minddataset_partition_num_samples_0(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test partition (using num_shards and shard_id) on MindDataset with num_samples=1
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    def partitions(num_shards):
        for partition_id in range(num_shards):
            data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                                      num_shards=num_shards,
                                      shard_id=partition_id, num_samples=1)

            assert data_set.get_dataset_size() == 1
            num_iter = 0
            for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
                logger.info("-------------- partition : {} ------------------------".format(partition_id))
                logger.info("-------------- item[file_name]: {}-----------------------".format(item["file_name"]))
                logger.info("-------------- item[label]: {} -----------------------".format(item["label"]))
                num_iter += 1
        return num_iter

    assert partitions(4) == 1
    assert partitions(5) == 1
    assert partitions(9) == 1


def test_cv_minddataset_partition_num_samples_1(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test partition (using num_shards and shard_id) on MindDataset
        with num_samples > 1 but num_samples <= dataset size
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    def partitions(num_shards):
        for partition_id in range(num_shards):
            data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                                      num_shards=num_shards,
                                      shard_id=partition_id, num_samples=2)

            assert data_set.get_dataset_size() == 2
            num_iter = 0
            for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
                logger.info("-------------- partition : {} ------------------------".format(partition_id))
                logger.info("-------------- item[file_name]: {}-----------------------".format(item["file_name"]))
                logger.info("-------------- item[label]: {} -----------------------".format(item["label"]))
                num_iter += 1
        return num_iter

    assert partitions(4) == 2
    assert partitions(5) == 2
    assert partitions(9) == 2


def test_cv_minddataset_partition_num_samples_2(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test partition (using num_shards and shard_id) on MindDataset
        with num_samples > 1 but num_samples > dataset size
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    def partitions(num_shards, expect):
        for partition_id in range(num_shards):
            data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                                      num_shards=num_shards,
                                      shard_id=partition_id, num_samples=3)

            assert data_set.get_dataset_size() == expect
            num_iter = 0
            for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
                logger.info("-------------- partition : {} ------------------------".format(partition_id))
                logger.info("-------------- item[file_name]: {}-----------------------".format(item["file_name"]))
                logger.info("-------------- item[label]: {} -----------------------".format(item["label"]))
                num_iter += 1
        return num_iter

    assert partitions(4, 3) == 3
    assert partitions(5, 2) == 2
    assert partitions(9, 2) == 2


def test_cv_minddataset_partition_num_samples_3(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test partition (using num_shards=1 and shard_id) on MindDataset
        with num_samples > 1 and num_samples = dataset size
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers, num_shards=1, shard_id=0, num_samples=5)

    assert data_set.get_dataset_size() == 5
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("-------------- item[file_name]: {}-----------------------".format(item["file_name"]))
        logger.info("-------------- item[label]: {} -----------------------".format(item["label"]))
        num_iter += 1

    assert num_iter == 5


def test_cv_minddataset_partition_tutorial_check_shuffle_result(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test partition (using num_shards=1 and shard_id) on MindDataset
        and check that the result is not shuffled
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    num_shards = 3
    epoch1 = []
    epoch2 = []
    epoch3 = []
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    for partition_id in range(num_shards):
        data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                                  num_shards=num_shards, shard_id=partition_id)

        data_set = data_set.repeat(3)

        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            logger.info("-------------- partition : {} ------------------------".format(partition_id))
            logger.info("-------------- item[file_name]: {}-----------------------".format(item["file_name"]))
            logger.info("-------------- item[label]: {} -----------------------".format(item["label"]))
            num_iter += 1
            if num_iter <= 4:
                epoch1.append(item["file_name"])  # save epoch 1 list
            elif num_iter <= 8:
                epoch2.append(item["file_name"])  # save epoch 2 list
            else:
                epoch3.append(item["file_name"])  # save epoch 3 list
        assert num_iter == 12
        assert len(epoch1) == 4
        assert len(epoch2) == 4
        assert len(epoch3) == 4
        assert epoch1 not in (epoch2, epoch3)
        assert epoch2 not in (epoch1, epoch3)
        assert epoch3 not in (epoch1, epoch2)
        epoch1 = []
        epoch2 = []
        epoch3 = []


def test_cv_minddataset_partition_tutorial_check_whole_reshuffle_result_per_epoch(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test partition (using num_shards=1 and shard_id) on MindDataset
        and check that the whole result under multiple epochs is not shuffled
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    num_shards = 3
    epoch_result = [[["", "", "", ""], ["", "", "", ""], ["", "", "", ""]],  # save partition 0 result
                    [["", "", "", ""], ["", "", "", ""], ["", "", "", ""]],  # save partition 1 result
                    [["", "", "", ""], ["", "", "", ""], ["", "", "", ""]]]  # svae partition 2 result

    for partition_id in range(num_shards):
        data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                                  num_shards=num_shards, shard_id=partition_id)

        data_set = data_set.repeat(3)

        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            logger.info("-------------- partition : {} ------------------------".format(partition_id))
            logger.info("-------------- item[file_name]: {}-----------------------".format(item["file_name"]))
            logger.info("-------------- item[label]: {} -----------------------".format(item["label"]))
            # total 3 partition, 4 result per epoch, total 12 result
            epoch_result[partition_id][int(num_iter / 4)][num_iter % 4] = item["file_name"]  # save epoch result
            num_iter += 1
        assert num_iter == 12
        assert epoch_result[partition_id][0] not in (epoch_result[partition_id][1], epoch_result[partition_id][2])
        assert epoch_result[partition_id][1] not in (epoch_result[partition_id][0], epoch_result[partition_id][2])
        assert epoch_result[partition_id][2] not in (epoch_result[partition_id][1], epoch_result[partition_id][0])
        epoch_result[partition_id][0].sort()
        epoch_result[partition_id][1].sort()
        epoch_result[partition_id][2].sort()
        assert epoch_result[partition_id][0] != epoch_result[partition_id][1]
        assert epoch_result[partition_id][1] != epoch_result[partition_id][2]
        assert epoch_result[partition_id][2] != epoch_result[partition_id][0]


def test_cv_minddataset_check_shuffle_result(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test read on MindDataset after Repeat op is applied and check that the result is not shuffled
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    original_seed = config_get_set_seed(54321)
    epoch1 = []
    epoch2 = []
    epoch3 = []

    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers)
    data_set = data_set.repeat(3)

    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("-------------- item[file_name]: {}-----------------------".format(item["file_name"]))
        logger.info("-------------- item[label]: {} -----------------------".format(item["label"]))
        num_iter += 1
        if num_iter <= 10:
            epoch1.append(item["file_name"])  # save epoch 1 list
        elif num_iter <= 20:
            epoch2.append(item["file_name"])  # save epoch 2 list
        else:
            epoch3.append(item["file_name"])  # save epoch 3 list
    assert num_iter == 30
    assert len(epoch1) == 10
    assert len(epoch2) == 10
    assert len(epoch3) == 10
    assert epoch1 not in (epoch2, epoch3)
    assert epoch2 not in (epoch1, epoch3)
    assert epoch3 not in (epoch1, epoch2)

    epoch1_new_dataset = []
    epoch2_new_dataset = []
    epoch3_new_dataset = []

    data_set2 = ds.MindDataset(file_name + "0", columns_list, num_readers)
    data_set2 = data_set2.repeat(3)

    num_iter = 0
    for item in data_set2.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("-------------- item[file_name]: {}-----------------------".format(item["file_name"]))
        logger.info("-------------- item[label]: {} -----------------------".format(item["label"]))
        num_iter += 1
        if num_iter <= 10:
            epoch1_new_dataset.append(item["file_name"])  # save epoch 1 list
        elif num_iter <= 20:
            epoch2_new_dataset.append(item["file_name"])  # save epoch 2 list
        else:
            epoch3_new_dataset.append(item["file_name"])  # save epoch 3 list
    assert num_iter == 30
    assert len(epoch1_new_dataset) == 10
    assert len(epoch2_new_dataset) == 10
    assert len(epoch3_new_dataset) == 10
    assert epoch1_new_dataset not in (epoch2_new_dataset, epoch3_new_dataset)
    assert epoch2_new_dataset not in (epoch1_new_dataset, epoch3_new_dataset)
    assert epoch3_new_dataset not in (epoch1_new_dataset, epoch2_new_dataset)

    assert epoch1 == epoch1_new_dataset
    assert epoch2 == epoch2_new_dataset
    assert epoch3 == epoch3_new_dataset

    ds.config.set_seed(original_seed)

    original_seed = config_get_set_seed(12345)
    epoch1_new_dataset2 = []
    epoch2_new_dataset2 = []
    epoch3_new_dataset2 = []

    data_set3 = ds.MindDataset(file_name + "0", columns_list, num_readers)
    data_set3 = data_set3.repeat(3)

    num_iter = 0
    for item in data_set3.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("-------------- item[file_name]: {}-----------------------".format(item["file_name"]))
        logger.info("-------------- item[label]: {} -----------------------".format(item["label"]))
        num_iter += 1
        if num_iter <= 10:
            epoch1_new_dataset2.append(item["file_name"])  # save epoch 1 list
        elif num_iter <= 20:
            epoch2_new_dataset2.append(item["file_name"])  # save epoch 2 list
        else:
            epoch3_new_dataset2.append(item["file_name"])  # save epoch 3 list
    assert num_iter == 30
    assert len(epoch1_new_dataset2) == 10
    assert len(epoch2_new_dataset2) == 10
    assert len(epoch3_new_dataset2) == 10
    assert epoch1_new_dataset2 not in (epoch2_new_dataset2, epoch3_new_dataset2)
    assert epoch2_new_dataset2 not in (epoch1_new_dataset2, epoch3_new_dataset2)
    assert epoch3_new_dataset2 not in (epoch1_new_dataset2, epoch2_new_dataset2)

    assert epoch1 != epoch1_new_dataset2
    assert epoch2 != epoch2_new_dataset2
    assert epoch3 != epoch3_new_dataset2

    ds.config.set_seed(original_seed)


def test_cv_minddataset_dataset_size(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test get_dataset_size on MindDataset
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers)
    assert data_set.get_dataset_size() == 10
    repeat_num = 2
    data_set = data_set.repeat(repeat_num)
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- get dataset size {} -----------------".format(num_iter))
        logger.info(
            "-------------- item[label]: {} ---------------------".format(item["label"]))
        logger.info(
            "-------------- item[data]: {} ----------------------".format(item["data"]))
        num_iter += 1
    assert num_iter == 20
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                              num_shards=4, shard_id=3)
    assert data_set.get_dataset_size() == 3


def test_cv_minddataset_repeat_reshuffle(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test read on MindDataset where after multiple Map ops and repeat op the result is not shuffled
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers)
    decode_op = vision.Decode()
    data_set = data_set.map(
        input_columns=["data"], operations=decode_op, num_parallel_workers=2)
    resize_op = vision.Resize((32, 32), interpolation=Inter.LINEAR)
    data_set = data_set.map(operations=resize_op, input_columns="data",
                            num_parallel_workers=2)
    data_set = data_set.batch(2)
    data_set = data_set.repeat(2)
    num_iter = 0
    labels = []
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- get dataset size {} -----------------".format(num_iter))
        logger.info(
            "-------------- item[label]: {} ---------------------".format(item["label"]))
        logger.info(
            "-------------- item[data]: {} ----------------------".format(item["data"]))
        num_iter += 1
        labels.append(item["label"])
    assert num_iter == 10
    logger.info("repeat shuffle: {}".format(labels))
    assert len(labels) == 10
    assert labels[0:5] == labels[0:5]
    assert labels[0:5] != labels[5:5]


def test_cv_minddataset_batch_size_larger_than_records(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test MindDataset when batch_size in Batch op is larger than records
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers)
    decode_op = vision.Decode()
    data_set = data_set.map(
        input_columns=["data"], operations=decode_op, num_parallel_workers=2)
    resize_op = vision.Resize((32, 32), interpolation=Inter.LINEAR)
    data_set = data_set.map(operations=resize_op, input_columns="data",
                            num_parallel_workers=2)
    data_set = data_set.batch(32, drop_remainder=True)
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- get dataset size {} -----------------".format(num_iter))
        logger.info(
            "-------------- item[label]: {} ---------------------".format(item["label"]))
        logger.info(
            "-------------- item[data]: {} ----------------------".format(item["data"]))
        num_iter += 1
    assert num_iter == 0


def test_cv_minddataset_issue_888(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test MindDataset by applying Shuffle op followed by Repeat op
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "label"]
    num_readers = 2
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers, shuffle=False, num_shards=5, shard_id=1)
    data_set = data_set.shuffle(2)
    data_set = data_set.repeat(9)
    num_iter = 0
    for _ in data_set.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 18


def test_cv_minddataset_reader_file_list(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test read on MindDataset using list of files
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset([file_name + str(x)
                               for x in range(FILES_NUM)], columns_list, num_readers)
    assert data_set.get_dataset_size() == 10
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info(
            "-------------- len(item[data]): {} ------------------------".format(len(item["data"])))
        logger.info(
            "-------------- item[data]: {} -----------------------------".format(item["data"]))
        logger.info(
            "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1
    assert num_iter == 10


def test_cv_minddataset_reader_one_partition(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test read on MindDataset using list of one file
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset([file_name + "0"], columns_list, num_readers)
    assert data_set.get_dataset_size() < 10
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info(
            "-------------- len(item[data]): {} ------------------------".format(len(item["data"])))
        logger.info(
            "-------------- item[data]: {} -----------------------------".format(item["data"]))
        logger.info(
            "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1
    assert num_iter < 10


def test_cv_minddataset_reader_two_dataset(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test read on MindDataset using 2 datasets that are written with FileWriter
    Expectation: Output is equal to the expected output
    """
    cv1_file_name = "../data/mindrecord/test_cv_minddataset_reader_two_dataset_1.mindrecord"
    cv2_file_name = "../data/mindrecord/test_cv_minddataset_reader_two_dataset_2.mindrecord"
    try:
        if os.path.exists(cv1_file_name):
            os.remove(cv1_file_name)
        if os.path.exists("{}.db".format(cv1_file_name)):
            os.remove("{}.db".format(cv1_file_name))
        if os.path.exists(cv2_file_name):
            os.remove(cv2_file_name)
        if os.path.exists("{}.db".format(cv2_file_name)):
            os.remove("{}.db".format(cv2_file_name))
        writer = FileWriter(cv1_file_name, 1)
        data = get_data(CV_DIR_NAME)
        cv_schema_json = {"id": {"type": "int32"},
                          "file_name": {"type": "string"},
                          "label": {"type": "int32"},
                          "data": {"type": "bytes"}}
        writer.add_schema(cv_schema_json, "CV1_schema")
        writer.add_index(["file_name", "label"])
        writer.write_raw_data(data)
        writer.commit()

        writer = FileWriter(cv2_file_name, 1)
        data = get_data(CV_DIR_NAME)
        cv_schema_json = {"id": {"type": "int32"},
                          "file_name": {"type": "string"},
                          "label": {"type": "int32"},
                          "data": {"type": "bytes"}}
        writer.add_schema(cv_schema_json, "CV2_schema")
        writer.add_index(["file_name", "label"])
        writer.write_raw_data(data)
        writer.commit()
        columns_list = ["data", "file_name", "label"]
        num_readers = 4
        file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
        data_set = ds.MindDataset([file_name + str(x) for x in range(FILES_NUM)] + [cv1_file_name, cv2_file_name],
                                  columns_list, num_readers)
        assert data_set.get_dataset_size() == 30
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            logger.info(
                "-------------- cv reader basic: {} ------------------------".format(num_iter))
            logger.info(
                "-------------- len(item[data]): {} ------------------------".format(len(item["data"])))
            logger.info(
                "-------------- item[data]: {} -----------------------------".format(item["data"]))
            logger.info(
                "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
            logger.info(
                "-------------- item[label]: {} ----------------------------".format(item["label"]))
            num_iter += 1
        assert num_iter == 30
    except Exception as error:
        if os.path.exists(cv1_file_name):
            os.remove(cv1_file_name)
        if os.path.exists("{}.db".format(cv1_file_name)):
            os.remove("{}.db".format(cv1_file_name))
        if os.path.exists(cv2_file_name):
            os.remove(cv2_file_name)
        if os.path.exists("{}.db".format(cv2_file_name)):
            os.remove("{}.db".format(cv2_file_name))
        raise error
    else:
        if os.path.exists(cv1_file_name):
            os.remove(cv1_file_name)
        if os.path.exists("{}.db".format(cv1_file_name)):
            os.remove("{}.db".format(cv1_file_name))
        if os.path.exists(cv2_file_name):
            os.remove(cv2_file_name)
        if os.path.exists("{}.db".format(cv2_file_name)):
            os.remove("{}.db".format(cv2_file_name))


def test_cv_minddataset_reader_two_dataset_partition(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test read on MindDataset using two datasets that are partitioned into two lists
    Expectation: Output is equal to the expected output
    """
    cv1_file_name = "../data/mindrecord/test_cv_minddataset_reader_two_dataset_partition_1"
    paths = ["{}{}".format(cv1_file_name, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    try:
        for x in paths:
            if os.path.exists("{}".format(x)):
                os.remove("{}".format(x))
            if os.path.exists("{}.db".format(x)):
                os.remove("{}.db".format(x))
        writer = FileWriter(cv1_file_name, FILES_NUM)
        data = get_data(CV_DIR_NAME)
        cv_schema_json = {"id": {"type": "int32"},
                          "file_name": {"type": "string"},
                          "label": {"type": "int32"},
                          "data": {"type": "bytes"}}
        writer.add_schema(cv_schema_json, "CV1_schema")
        writer.add_index(["file_name", "label"])
        writer.write_raw_data(data)
        writer.commit()

        columns_list = ["data", "file_name", "label"]
        num_readers = 4
        file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
        data_set = ds.MindDataset([file_name + str(x) for x in range(2)] +
                                  [cv1_file_name + str(x) for x in range(2, 4)],
                                  columns_list, num_readers)
        assert data_set.get_dataset_size() < 20
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            logger.info(
                "-------------- cv reader basic: {} ------------------------".format(num_iter))
            logger.info(
                "-------------- len(item[data]): {} ------------------------".format(len(item["data"])))
            logger.info(
                "-------------- item[data]: {} -----------------------------".format(item["data"]))
            logger.info(
                "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
            logger.info(
                "-------------- item[label]: {} ----------------------------".format(item["label"]))
            num_iter += 1
        assert num_iter < 20
    except Exception as error:
        for x in paths:
            os.remove("{}".format(x))
            os.remove("{}.db".format(x))
        raise error
    else:
        for x in paths:
            os.remove("{}".format(x))
            os.remove("{}.db".format(x))


def test_cv_minddataset_reader_basic_tutorial(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test basic read on MindDataset tutorial
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers)
    assert data_set.get_dataset_size() == 10
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info(
            "-------------- len(item[data]): {} ------------------------".format(len(item["data"])))
        logger.info(
            "-------------- item[data]: {} -----------------------------".format(item["data"]))
        logger.info(
            "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1
    assert num_iter == 10


def test_nlp_minddataset_reader_basic_tutorial(add_and_remove_nlp_file):
    """
    Feature: MindDataset
    Description: Test basic read on NLP MindDataset tutorial
    Expectation: Output is equal to the expected output
    """
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset(file_name + "0", None, num_readers)
    assert data_set.get_dataset_size() == 10
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info(
            "-------------- num_iter: {} ------------------------".format(num_iter))
        logger.info(
            "-------------- item[id]: {} ------------------------".format(item["id"]))
        logger.info(
            "-------------- item[rating]: {} --------------------".format(item["rating"]))
        logger.info("-------------- item[input_ids]: {}, shape: {} -----------------".format(
            item["input_ids"], item["input_ids"].shape))
        logger.info("-------------- item[input_mask]: {}, shape: {} -----------------".format(
            item["input_mask"], item["input_mask"].shape))
        logger.info("-------------- item[segment_ids]: {}, shape: {} -----------------".format(
            item["segment_ids"], item["segment_ids"].shape))
        assert item["input_ids"].shape == (50,)
        assert item["input_mask"].shape == (1, 50)
        assert item["segment_ids"].shape == (2, 25)
        num_iter += 1
    assert num_iter == 10


def test_cv_minddataset_reader_basic_tutorial_5_epoch(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test basic read on MindDataset tutorial with 5 epochs
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers)
    assert data_set.get_dataset_size() == 10
    for _ in range(5):
        num_iter = 0
        for data in data_set.create_tuple_iterator(num_epochs=1, output_numpy=True):
            logger.info("data is {}".format(data))
            num_iter += 1
        assert num_iter == 10

        data_set.reset()


def test_cv_minddataset_reader_basic_tutorial_5_epoch_with_batch(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test basic read on MindDataset tutorial with 5 epochs after Batch op
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers)

    resize_height = 32
    resize_width = 32

    # define map operations
    decode_op = vision.Decode()
    resize_op = vision.Resize((resize_height, resize_width))

    data_set = data_set.map(input_columns=["data"], operations=decode_op, num_parallel_workers=4)
    data_set = data_set.map(input_columns=["data"], operations=resize_op, num_parallel_workers=4)

    data_set = data_set.batch(2)
    assert data_set.get_dataset_size() == 5
    for _ in range(5):
        num_iter = 0
        for data in data_set.create_tuple_iterator(num_epochs=1, output_numpy=True):
            logger.info("data is {}".format(data))
            num_iter += 1
        assert num_iter == 5

        data_set.reset()


def test_cv_minddataset_reader_no_columns(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test read on MindDataset with no columns_list
    Expectation: Output is equal to the expected output
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset(file_name + "0")
    assert data_set.get_dataset_size() == 10
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info(
            "-------------- len(item[data]): {} ------------------------".format(len(item["data"])))
        logger.info(
            "-------------- item[data]: {} -----------------------------".format(item["data"]))
        logger.info(
            "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1
    assert num_iter == 10


def test_cv_minddataset_reader_repeat_tutorial(add_and_remove_cv_file):
    """
    Feature: MindDataset
    Description: Test read on MindDataset after Repeat op is applied on the dataset
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers)
    repeat_num = 2
    data_set = data_set.repeat(repeat_num)
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info(
            "-------------- repeat two test {} ------------------------".format(num_iter))
        logger.info(
            "-------------- len(item[data]): {} -----------------------".format(len(item["data"])))
        logger.info(
            "-------------- item[data]: {} ----------------------------".format(item["data"]))
        logger.info(
            "-------------- item[file_name]: {} -----------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ---------------------------".format(item["label"]))
        num_iter += 1
    assert num_iter == 20


def get_data(dir_name):
    """
    usage: get data from imagenet dataset
    params:
    dir_name: directory containing folder images and annotation information

    """
    if not os.path.isdir(dir_name):
        raise IOError("Directory {} not exists".format(dir_name))
    img_dir = os.path.join(dir_name, "images")
    ann_file = os.path.join(dir_name, "annotation.txt")
    with open(ann_file, "r") as file_reader:
        lines = file_reader.readlines()

    data_list = []
    for i, line in enumerate(lines):
        try:
            filename, label = line.split(",")
            label = label.strip("\n")
            with open(os.path.join(img_dir, filename), "rb") as file_reader:
                img = file_reader.read()
            data_json = {"id": i,
                         "file_name": filename,
                         "data": img,
                         "label": int(label)}
            data_list.append(data_json)
        except FileNotFoundError:
            continue
    return data_list


def get_multi_bytes_data(file_name, bytes_num=3):
    """
    Return raw data of multi-bytes dataset.

    Args:
        file_name (str): String of multi-bytes dataset's path.
        bytes_num (int): Number of bytes fields.

    Returns:
       List
    """
    if not os.path.exists(file_name):
        raise IOError("map file {} not exists".format(file_name))
    dir_name = os.path.dirname(file_name)
    with open(file_name, "r") as file_reader:
        lines = file_reader.readlines()
    data_list = []
    row_num = 0
    for line in lines:
        try:
            img10_path = line.strip('\n').split(" ")
            img5 = []
            for path in img10_path[:bytes_num]:
                with open(os.path.join(dir_name, path), "rb") as file_reader:
                    img5 += [file_reader.read()]
            data_json = {"image_{}".format(i): img5[i]
                         for i in range(len(img5))}
            data_json.update({"id": row_num})
            row_num += 1
            data_list.append(data_json)
        except FileNotFoundError:
            continue
    return data_list


def get_mkv_data(dir_name):
    """
    Return raw data of Vehicle_and_Person dataset.

    Args:
        dir_name (str): String of Vehicle_and_Person dataset's path.

    Returns:
        List
    """
    if not os.path.isdir(dir_name):
        raise IOError("Directory {} not exists".format(dir_name))
    img_dir = os.path.join(dir_name, "Image")
    label_dir = os.path.join(dir_name, "prelabel")

    data_list = []
    file_list = os.listdir(label_dir)

    index = 1
    for item in file_list:
        if os.path.splitext(item)[1] == '.json':
            file_path = os.path.join(label_dir, item)

            image_name = ''.join([os.path.splitext(item)[0], ".jpg"])
            image_path = os.path.join(img_dir, image_name)

            with open(file_path, "r") as load_f:
                load_dict = json.load(load_f)

            if os.path.exists(image_path):
                with open(image_path, "rb") as file_reader:
                    img = file_reader.read()
                data_json = {"file_name": image_name,
                             "prelabel": str(load_dict),
                             "data": img,
                             "id": index}
                data_list.append(data_json)
            index += 1
    logger.info('{} images are missing'.format(
        len(file_list) - len(data_list)))
    return data_list


def get_nlp_data(dir_name, vocab_file, num):
    """
    Return raw data of aclImdb dataset.

    Args:
        dir_name (str): String of aclImdb dataset's path.
        vocab_file (str): String of dictionary's path.
        num (int): Number of sample.

    Returns:
        List
    """
    if not os.path.isdir(dir_name):
        raise IOError("Directory {} not exists".format(dir_name))
    for root, _, files in os.walk(dir_name):
        for index, file_name_extension in enumerate(files):
            if index < num:
                file_path = os.path.join(root, file_name_extension)
                file_name, _ = file_name_extension.split('.', 1)
                id_, rating = file_name.split('_', 1)
                with open(file_path, 'r') as f:
                    raw_content = f.read()

                dictionary = load_vocab(vocab_file)
                vectors = [dictionary.get('[CLS]')]
                vectors += [dictionary.get(i) if i in dictionary
                            else dictionary.get('[UNK]')
                            for i in re.findall(r"[\w']+|[{}]"
                                                .format(string.punctuation),
                                                raw_content)]
                vectors += [dictionary.get('[SEP]')]
                input_, mask, segment = inputs(vectors)
                input_ids = np.reshape(np.array(input_), [-1])
                input_mask = np.reshape(np.array(mask), [1, -1])
                segment_ids = np.reshape(np.array(segment), [2, -1])
                data = {
                    "label": 1,
                    "id": id_,
                    "rating": float(rating),
                    "input_ids": input_ids,
                    "input_mask": input_mask,
                    "segment_ids": segment_ids
                }
                yield data


def convert_to_uni(text):
    if isinstance(text, str):
        return text
    if isinstance(text, bytes):
        return text.decode('utf-8', 'ignore')
    raise Exception("The type %s does not convert!" % type(text))


def load_vocab(vocab_file):
    """load vocabulary to translate statement."""
    vocab = collections.OrderedDict()
    vocab.setdefault('blank', 2)
    index = 0
    with open(vocab_file) as reader:
        while True:
            tmp = reader.readline()
            if not tmp:
                break
            token = convert_to_uni(tmp)
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def inputs(vectors, maxlen=50):
    length = len(vectors)
    if length > maxlen:
        return vectors[0:maxlen], [1] * maxlen, [0] * maxlen
    input_ = vectors + [0] * (maxlen - length)
    mask = [1] * length + [0] * (maxlen - length)
    segment = [0] * maxlen
    return input_, mask, segment


def test_write_with_multi_bytes_and_array_and_read_by_MindDataset():
    """
    Feature: MindDataset
    Description: Test write multiple bytes and arrays using FileWriter and read them by MindDataset
    Expectation: Output is equal to the expected output
    """
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    try:
        if os.path.exists("{}".format(mindrecord_file_name)):
            os.remove("{}".format(mindrecord_file_name))
        if os.path.exists("{}.db".format(mindrecord_file_name)):
            os.remove("{}.db".format(mindrecord_file_name))
        data = [{"file_name": "001.jpg", "label": 4,
                 "image1": bytes("image1 bytes abc", encoding='UTF-8'),
                 "image2": bytes("image1 bytes def", encoding='UTF-8'),
                 "source_sos_ids": np.array([1, 2, 3, 4, 5], dtype=np.int64),
                 "source_sos_mask": np.array([6, 7, 8, 9, 10, 11, 12], dtype=np.int64),
                 "image3": bytes("image1 bytes ghi", encoding='UTF-8'),
                 "image4": bytes("image1 bytes jkl", encoding='UTF-8'),
                 "image5": bytes("image1 bytes mno", encoding='UTF-8'),
                 "target_sos_ids": np.array([28, 29, 30, 31, 32], dtype=np.int64),
                 "target_sos_mask": np.array([33, 34, 35, 36, 37, 38], dtype=np.int64),
                 "target_eos_ids": np.array([39, 40, 41, 42, 43, 44, 45, 46, 47], dtype=np.int64),
                 "target_eos_mask": np.array([48, 49, 50, 51], dtype=np.int64)},
                {"file_name": "002.jpg", "label": 5,
                 "image1": bytes("image2 bytes abc", encoding='UTF-8'),
                 "image2": bytes("image2 bytes def", encoding='UTF-8'),
                 "image3": bytes("image2 bytes ghi", encoding='UTF-8'),
                 "image4": bytes("image2 bytes jkl", encoding='UTF-8'),
                 "image5": bytes("image2 bytes mno", encoding='UTF-8'),
                 "source_sos_ids": np.array([11, 2, 3, 4, 5], dtype=np.int64),
                 "source_sos_mask": np.array([16, 7, 8, 9, 10, 11, 12], dtype=np.int64),
                 "target_sos_ids": np.array([128, 29, 30, 31, 32], dtype=np.int64),
                 "target_sos_mask": np.array([133, 34, 35, 36, 37, 38], dtype=np.int64),
                 "target_eos_ids": np.array([139, 40, 41, 42, 43, 44, 45, 46, 47], dtype=np.int64),
                 "target_eos_mask": np.array([148, 49, 50, 51], dtype=np.int64)},
                {"file_name": "003.jpg", "label": 6,
                 "source_sos_ids": np.array([21, 2, 3, 4, 5], dtype=np.int64),
                 "source_sos_mask": np.array([26, 7, 8, 9, 10, 11, 12], dtype=np.int64),
                 "target_sos_ids": np.array([228, 29, 30, 31, 32], dtype=np.int64),
                 "target_sos_mask": np.array([233, 34, 35, 36, 37, 38], dtype=np.int64),
                 "target_eos_ids": np.array([239, 40, 41, 42, 43, 44, 45, 46, 47], dtype=np.int64),
                 "image1": bytes("image3 bytes abc", encoding='UTF-8'),
                 "image2": bytes("image3 bytes def", encoding='UTF-8'),
                 "image3": bytes("image3 bytes ghi", encoding='UTF-8'),
                 "image4": bytes("image3 bytes jkl", encoding='UTF-8'),
                 "image5": bytes("image3 bytes mno", encoding='UTF-8'),
                 "target_eos_mask": np.array([248, 49, 50, 51], dtype=np.int64)},
                {"file_name": "004.jpg", "label": 7,
                 "source_sos_ids": np.array([31, 2, 3, 4, 5], dtype=np.int64),
                 "source_sos_mask": np.array([36, 7, 8, 9, 10, 11, 12], dtype=np.int64),
                 "image1": bytes("image4 bytes abc", encoding='UTF-8'),
                 "image2": bytes("image4 bytes def", encoding='UTF-8'),
                 "image3": bytes("image4 bytes ghi", encoding='UTF-8'),
                 "image4": bytes("image4 bytes jkl", encoding='UTF-8'),
                 "image5": bytes("image4 bytes mno", encoding='UTF-8'),
                 "target_sos_ids": np.array([328, 29, 30, 31, 32], dtype=np.int64),
                 "target_sos_mask": np.array([333, 34, 35, 36, 37, 38], dtype=np.int64),
                 "target_eos_ids": np.array([339, 40, 41, 42, 43, 44, 45, 46, 47], dtype=np.int64),
                 "target_eos_mask": np.array([348, 49, 50, 51], dtype=np.int64)},
                {"file_name": "005.jpg", "label": 8,
                 "source_sos_ids": np.array([41, 2, 3, 4, 5], dtype=np.int64),
                 "source_sos_mask": np.array([46, 7, 8, 9, 10, 11, 12], dtype=np.int64),
                 "target_sos_ids": np.array([428, 29, 30, 31, 32], dtype=np.int64),
                 "target_sos_mask": np.array([433, 34, 35, 36, 37, 38], dtype=np.int64),
                 "image1": bytes("image5 bytes abc", encoding='UTF-8'),
                 "image2": bytes("image5 bytes def", encoding='UTF-8'),
                 "image3": bytes("image5 bytes ghi", encoding='UTF-8'),
                 "image4": bytes("image5 bytes jkl", encoding='UTF-8'),
                 "image5": bytes("image5 bytes mno", encoding='UTF-8'),
                 "target_eos_ids": np.array([439, 40, 41, 42, 43, 44, 45, 46, 47], dtype=np.int64),
                 "target_eos_mask": np.array([448, 49, 50, 51], dtype=np.int64)},
                {"file_name": "006.jpg", "label": 9,
                 "source_sos_ids": np.array([51, 2, 3, 4, 5], dtype=np.int64),
                 "source_sos_mask": np.array([56, 7, 8, 9, 10, 11, 12], dtype=np.int64),
                 "target_sos_ids": np.array([528, 29, 30, 31, 32], dtype=np.int64),
                 "image1": bytes("image6 bytes abc", encoding='UTF-8'),
                 "image2": bytes("image6 bytes def", encoding='UTF-8'),
                 "image3": bytes("image6 bytes ghi", encoding='UTF-8'),
                 "image4": bytes("image6 bytes jkl", encoding='UTF-8'),
                 "image5": bytes("image6 bytes mno", encoding='UTF-8'),
                 "target_sos_mask": np.array([533, 34, 35, 36, 37, 38], dtype=np.int64),
                 "target_eos_ids": np.array([539, 40, 41, 42, 43, 44, 45, 46, 47], dtype=np.int64),
                 "target_eos_mask": np.array([548, 49, 50, 51], dtype=np.int64)}
                ]

        writer = FileWriter(mindrecord_file_name)
        schema = {"file_name": {"type": "string"},
                  "image1": {"type": "bytes"},
                  "image2": {"type": "bytes"},
                  "source_sos_ids": {"type": "int64", "shape": [-1]},
                  "source_sos_mask": {"type": "int64", "shape": [-1]},
                  "image3": {"type": "bytes"},
                  "image4": {"type": "bytes"},
                  "image5": {"type": "bytes"},
                  "target_sos_ids": {"type": "int64", "shape": [-1]},
                  "target_sos_mask": {"type": "int64", "shape": [-1]},
                  "target_eos_ids": {"type": "int64", "shape": [-1]},
                  "target_eos_mask": {"type": "int64", "shape": [-1]},
                  "label": {"type": "int32"}}
        writer.add_schema(schema, "data is so cool")
        writer.write_raw_data(data)
        writer.commit()

        # change data value to list
        data_value_to_list = []
        for item in data:
            new_data = {}
            new_data['file_name'] = np.asarray(item["file_name"], dtype=np.str_)
            new_data['label'] = np.asarray(list([item["label"]]), dtype=np.int32)
            new_data['image1'] = np.asarray(list(item["image1"]), dtype=np.uint8)
            new_data['image2'] = np.asarray(list(item["image2"]), dtype=np.uint8)
            new_data['image3'] = np.asarray(list(item["image3"]), dtype=np.uint8)
            new_data['image4'] = np.asarray(list(item["image4"]), dtype=np.uint8)
            new_data['image5'] = np.asarray(list(item["image5"]), dtype=np.uint8)
            new_data['source_sos_ids'] = item["source_sos_ids"]
            new_data['source_sos_mask'] = item["source_sos_mask"]
            new_data['target_sos_ids'] = item["target_sos_ids"]
            new_data['target_sos_mask'] = item["target_sos_mask"]
            new_data['target_eos_ids'] = item["target_eos_ids"]
            new_data['target_eos_mask'] = item["target_eos_mask"]
            data_value_to_list.append(new_data)

        num_readers = 2
        data_set = ds.MindDataset(dataset_files=mindrecord_file_name,
                                  num_parallel_workers=num_readers,
                                  shuffle=False)
        assert data_set.get_dataset_size() == 6
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            assert len(item) == 13
            for field in item:
                if isinstance(item[field], np.ndarray):
                    assert (item[field] ==
                            data_value_to_list[num_iter][field]).all()
                else:
                    assert item[field] == data_value_to_list[num_iter][field]
            num_iter += 1
        assert num_iter == 6

        num_readers = 2
        data_set = ds.MindDataset(dataset_files=mindrecord_file_name,
                                  columns_list=["source_sos_ids",
                                                "source_sos_mask", "target_sos_ids"],
                                  num_parallel_workers=num_readers,
                                  shuffle=False)
        assert data_set.get_dataset_size() == 6
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            assert len(item) == 3
            for field in item:
                if isinstance(item[field], np.ndarray):
                    assert (item[field] == data[num_iter][field]).all()
                else:
                    assert item[field] == data[num_iter][field]
            num_iter += 1
        assert num_iter == 6

        num_readers = 1
        data_set = ds.MindDataset(dataset_files=mindrecord_file_name,
                                  columns_list=["image2", "source_sos_mask", "image3", "target_sos_ids"],
                                  num_parallel_workers=num_readers,
                                  shuffle=False)
        assert data_set.get_dataset_size() == 6
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            assert len(item) == 4
            for field in item:
                if isinstance(item[field], np.ndarray):
                    assert (item[field] ==
                            data_value_to_list[num_iter][field]).all()
                else:
                    assert item[field] == data_value_to_list[num_iter][field]
            num_iter += 1
        assert num_iter == 6

        num_readers = 3
        data_set = ds.MindDataset(dataset_files=mindrecord_file_name,
                                  columns_list=["target_sos_ids",
                                                "image4", "source_sos_ids"],
                                  num_parallel_workers=num_readers,
                                  shuffle=False)
        assert data_set.get_dataset_size() == 6
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            assert len(item) == 3
            for field in item:
                if isinstance(item[field], np.ndarray):
                    assert (item[field] ==
                            data_value_to_list[num_iter][field]).all()
                else:
                    assert item[field] == data_value_to_list[num_iter][field]
            num_iter += 1
        assert num_iter == 6

        num_readers = 3
        data_set = ds.MindDataset(dataset_files=mindrecord_file_name,
                                  columns_list=["target_sos_ids", "image5",
                                                "image4", "image3", "source_sos_ids"],
                                  num_parallel_workers=num_readers,
                                  shuffle=False)
        assert data_set.get_dataset_size() == 6
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            assert len(item) == 5
            for field in item:
                if isinstance(item[field], np.ndarray):
                    assert (item[field] ==
                            data_value_to_list[num_iter][field]).all()
                else:
                    assert item[field] == data_value_to_list[num_iter][field]
            num_iter += 1
        assert num_iter == 6

        num_readers = 1
        data_set = ds.MindDataset(dataset_files=mindrecord_file_name,
                                  columns_list=["target_eos_mask", "image5",
                                                "image2", "source_sos_mask", "label"],
                                  num_parallel_workers=num_readers,
                                  shuffle=False)
        assert data_set.get_dataset_size() == 6
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            assert len(item) == 5
            for field in item:
                if isinstance(item[field], np.ndarray):
                    assert (item[field] ==
                            data_value_to_list[num_iter][field]).all()
                else:
                    assert item[field] == data_value_to_list[num_iter][field]
            num_iter += 1
        assert num_iter == 6

        num_readers = 2
        data_set = ds.MindDataset(dataset_files=mindrecord_file_name,
                                  columns_list=["label", "target_eos_mask", "image1", "target_eos_ids",
                                                "source_sos_mask", "image2", "image4", "image3",
                                                "source_sos_ids", "image5", "file_name"],
                                  num_parallel_workers=num_readers,
                                  shuffle=False)
        assert data_set.get_dataset_size() == 6
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            assert len(item) == 11
            for field in item:
                if isinstance(item[field], np.ndarray):
                    assert (item[field] ==
                            data_value_to_list[num_iter][field]).all()
                else:
                    assert item[field] == data_value_to_list[num_iter][field]
            num_iter += 1
        assert num_iter == 6
    except Exception as error:
        os.remove("{}".format(mindrecord_file_name))
        os.remove("{}.db".format(mindrecord_file_name))
        raise error
    else:
        os.remove("{}".format(mindrecord_file_name))
        os.remove("{}.db".format(mindrecord_file_name))


def test_write_with_multi_bytes_and_MindDataset():
    """
    Feature: MindDataset
    Description: Test write multiple bytes using FileWriter and read them by MindDataset
    Expectation: Output is equal to the expected output
    """
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    try:
        data = [{"file_name": "001.jpg", "label": 43,
                 "image1": bytes("image1 bytes abc", encoding='UTF-8'),
                 "image2": bytes("image1 bytes def", encoding='UTF-8'),
                 "image3": bytes("image1 bytes ghi", encoding='UTF-8'),
                 "image4": bytes("image1 bytes jkl", encoding='UTF-8'),
                 "image5": bytes("image1 bytes mno", encoding='UTF-8')},
                {"file_name": "002.jpg", "label": 91,
                 "image1": bytes("image2 bytes abc", encoding='UTF-8'),
                 "image2": bytes("image2 bytes def", encoding='UTF-8'),
                 "image3": bytes("image2 bytes ghi", encoding='UTF-8'),
                 "image4": bytes("image2 bytes jkl", encoding='UTF-8'),
                 "image5": bytes("image2 bytes mno", encoding='UTF-8')},
                {"file_name": "003.jpg", "label": 61,
                 "image1": bytes("image3 bytes abc", encoding='UTF-8'),
                 "image2": bytes("image3 bytes def", encoding='UTF-8'),
                 "image3": bytes("image3 bytes ghi", encoding='UTF-8'),
                 "image4": bytes("image3 bytes jkl", encoding='UTF-8'),
                 "image5": bytes("image3 bytes mno", encoding='UTF-8')},
                {"file_name": "004.jpg", "label": 29,
                 "image1": bytes("image4 bytes abc", encoding='UTF-8'),
                 "image2": bytes("image4 bytes def", encoding='UTF-8'),
                 "image3": bytes("image4 bytes ghi", encoding='UTF-8'),
                 "image4": bytes("image4 bytes jkl", encoding='UTF-8'),
                 "image5": bytes("image4 bytes mno", encoding='UTF-8')},
                {"file_name": "005.jpg", "label": 78,
                 "image1": bytes("image5 bytes abc", encoding='UTF-8'),
                 "image2": bytes("image5 bytes def", encoding='UTF-8'),
                 "image3": bytes("image5 bytes ghi", encoding='UTF-8'),
                 "image4": bytes("image5 bytes jkl", encoding='UTF-8'),
                 "image5": bytes("image5 bytes mno", encoding='UTF-8')},
                {"file_name": "006.jpg", "label": 37,
                 "image1": bytes("image6 bytes abc", encoding='UTF-8'),
                 "image2": bytes("image6 bytes def", encoding='UTF-8'),
                 "image3": bytes("image6 bytes ghi", encoding='UTF-8'),
                 "image4": bytes("image6 bytes jkl", encoding='UTF-8'),
                 "image5": bytes("image6 bytes mno", encoding='UTF-8')}
                ]
        writer = FileWriter(mindrecord_file_name)
        schema = {"file_name": {"type": "string"},
                  "image1": {"type": "bytes"},
                  "image2": {"type": "bytes"},
                  "image3": {"type": "bytes"},
                  "label": {"type": "int32"},
                  "image4": {"type": "bytes"},
                  "image5": {"type": "bytes"}}
        writer.add_schema(schema, "data is so cool")
        writer.write_raw_data(data)
        writer.commit()

        # change data value to list
        data_value_to_list = []
        for item in data:
            new_data = {}
            new_data['file_name'] = np.asarray(item["file_name"], dtype=np.str_)
            new_data['label'] = np.asarray(list([item["label"]]), dtype=np.int32)
            new_data['image1'] = np.asarray(list(item["image1"]), dtype=np.uint8)
            new_data['image2'] = np.asarray(list(item["image2"]), dtype=np.uint8)
            new_data['image3'] = np.asarray(list(item["image3"]), dtype=np.uint8)
            new_data['image4'] = np.asarray(list(item["image4"]), dtype=np.uint8)
            new_data['image5'] = np.asarray(list(item["image5"]), dtype=np.uint8)
            data_value_to_list.append(new_data)

        num_readers = 2
        data_set = ds.MindDataset(dataset_files=mindrecord_file_name,
                                  num_parallel_workers=num_readers,
                                  shuffle=False)
        assert data_set.get_dataset_size() == 6
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            assert len(item) == 7
            for field in item:
                if isinstance(item[field], np.ndarray):
                    assert (item[field] ==
                            data_value_to_list[num_iter][field]).all()
                else:
                    assert item[field] == data_value_to_list[num_iter][field]
            num_iter += 1
        assert num_iter == 6

        num_readers = 2
        data_set = ds.MindDataset(dataset_files=mindrecord_file_name,
                                  columns_list=["image1", "image2", "image5"],
                                  num_parallel_workers=num_readers,
                                  shuffle=False)
        assert data_set.get_dataset_size() == 6
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            assert len(item) == 3
            for field in item:
                if isinstance(item[field], np.ndarray):
                    assert (item[field] ==
                            data_value_to_list[num_iter][field]).all()
                else:
                    assert item[field] == data_value_to_list[num_iter][field]
            num_iter += 1
        assert num_iter == 6

        num_readers = 2
        data_set = ds.MindDataset(dataset_files=mindrecord_file_name,
                                  columns_list=["image2", "image4"],
                                  num_parallel_workers=num_readers,
                                  shuffle=False)
        assert data_set.get_dataset_size() == 6
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            assert len(item) == 2
            for field in item:
                if isinstance(item[field], np.ndarray):
                    assert (item[field] ==
                            data_value_to_list[num_iter][field]).all()
                else:
                    assert item[field] == data_value_to_list[num_iter][field]
            num_iter += 1
        assert num_iter == 6

        num_readers = 2
        data_set = ds.MindDataset(dataset_files=mindrecord_file_name,
                                  columns_list=["image5", "image2"],
                                  num_parallel_workers=num_readers,
                                  shuffle=False)
        assert data_set.get_dataset_size() == 6
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            assert len(item) == 2
            for field in item:
                if isinstance(item[field], np.ndarray):
                    assert (item[field] ==
                            data_value_to_list[num_iter][field]).all()
                else:
                    assert item[field] == data_value_to_list[num_iter][field]
            num_iter += 1
        assert num_iter == 6

        num_readers = 2
        data_set = ds.MindDataset(dataset_files=mindrecord_file_name,
                                  columns_list=["image5", "image2", "label"],
                                  num_parallel_workers=num_readers,
                                  shuffle=False)
        assert data_set.get_dataset_size() == 6
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            assert len(item) == 3
            for field in item:
                if isinstance(item[field], np.ndarray):
                    assert (item[field] ==
                            data_value_to_list[num_iter][field]).all()
                else:
                    assert item[field] == data_value_to_list[num_iter][field]
            num_iter += 1
        assert num_iter == 6

        num_readers = 2
        data_set = ds.MindDataset(dataset_files=mindrecord_file_name,
                                  columns_list=["image4", "image5",
                                                "image2", "image3", "file_name"],
                                  num_parallel_workers=num_readers,
                                  shuffle=False)
        assert data_set.get_dataset_size() == 6
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            assert len(item) == 5
            for field in item:
                if isinstance(item[field], np.ndarray):
                    assert (item[field] ==
                            data_value_to_list[num_iter][field]).all()
                else:
                    assert item[field] == data_value_to_list[num_iter][field]
            num_iter += 1
        assert num_iter == 6
    except Exception as error:
        os.remove("{}".format(mindrecord_file_name))
        os.remove("{}.db".format(mindrecord_file_name))
        raise error
    else:
        os.remove("{}".format(mindrecord_file_name))
        os.remove("{}.db".format(mindrecord_file_name))


def test_write_with_multi_array_and_MindDataset():
    """
    Feature: MindDataset
    Description: Test write multiple arrays using FileWriter and read them by MindDataset
    Expectation: Output is equal to the expected output
    """
    mindrecord_file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    try:
        data = [{"source_sos_ids": np.array([1, 2, 3, 4, 5], dtype=np.int64),
                 "source_sos_mask": np.array([6, 7, 8, 9, 10, 11, 12], dtype=np.int64),
                 "source_eos_ids": np.array([13, 14, 15, 16, 17, 18], dtype=np.int64),
                 "source_eos_mask": np.array([19, 20, 21, 22, 23, 24, 25, 26, 27], dtype=np.int64),
                 "target_sos_ids": np.array([28, 29, 30, 31, 32], dtype=np.int64),
                 "target_sos_mask": np.array([33, 34, 35, 36, 37, 38], dtype=np.int64),
                 "target_eos_ids": np.array([39, 40, 41, 42, 43, 44, 45, 46, 47], dtype=np.int64),
                 "target_eos_mask": np.array([48, 49, 50, 51], dtype=np.int64)},
                {"source_sos_ids": np.array([11, 2, 3, 4, 5], dtype=np.int64),
                 "source_sos_mask": np.array([16, 7, 8, 9, 10, 11, 12], dtype=np.int64),
                 "source_eos_ids": np.array([113, 14, 15, 16, 17, 18], dtype=np.int64),
                 "source_eos_mask": np.array([119, 20, 21, 22, 23, 24, 25, 26, 27], dtype=np.int64),
                 "target_sos_ids": np.array([128, 29, 30, 31, 32], dtype=np.int64),
                 "target_sos_mask": np.array([133, 34, 35, 36, 37, 38], dtype=np.int64),
                 "target_eos_ids": np.array([139, 40, 41, 42, 43, 44, 45, 46, 47], dtype=np.int64),
                 "target_eos_mask": np.array([148, 49, 50, 51], dtype=np.int64)},
                {"source_sos_ids": np.array([21, 2, 3, 4, 5], dtype=np.int64),
                 "source_sos_mask": np.array([26, 7, 8, 9, 10, 11, 12], dtype=np.int64),
                 "source_eos_ids": np.array([213, 14, 15, 16, 17, 18], dtype=np.int64),
                 "source_eos_mask": np.array([219, 20, 21, 22, 23, 24, 25, 26, 27], dtype=np.int64),
                 "target_sos_ids": np.array([228, 29, 30, 31, 32], dtype=np.int64),
                 "target_sos_mask": np.array([233, 34, 35, 36, 37, 38], dtype=np.int64),
                 "target_eos_ids": np.array([239, 40, 41, 42, 43, 44, 45, 46, 47], dtype=np.int64),
                 "target_eos_mask": np.array([248, 49, 50, 51], dtype=np.int64)},
                {"source_sos_ids": np.array([31, 2, 3, 4, 5], dtype=np.int64),
                 "source_sos_mask": np.array([36, 7, 8, 9, 10, 11, 12], dtype=np.int64),
                 "source_eos_ids": np.array([313, 14, 15, 16, 17, 18], dtype=np.int64),
                 "source_eos_mask": np.array([319, 20, 21, 22, 23, 24, 25, 26, 27], dtype=np.int64),
                 "target_sos_ids": np.array([328, 29, 30, 31, 32], dtype=np.int64),
                 "target_sos_mask": np.array([333, 34, 35, 36, 37, 38], dtype=np.int64),
                 "target_eos_ids": np.array([339, 40, 41, 42, 43, 44, 45, 46, 47], dtype=np.int64),
                 "target_eos_mask": np.array([348, 49, 50, 51], dtype=np.int64)},
                {"source_sos_ids": np.array([41, 2, 3, 4, 5], dtype=np.int64),
                 "source_sos_mask": np.array([46, 7, 8, 9, 10, 11, 12], dtype=np.int64),
                 "source_eos_ids": np.array([413, 14, 15, 16, 17, 18], dtype=np.int64),
                 "source_eos_mask": np.array([419, 20, 21, 22, 23, 24, 25, 26, 27], dtype=np.int64),
                 "target_sos_ids": np.array([428, 29, 30, 31, 32], dtype=np.int64),
                 "target_sos_mask": np.array([433, 34, 35, 36, 37, 38], dtype=np.int64),
                 "target_eos_ids": np.array([439, 40, 41, 42, 43, 44, 45, 46, 47], dtype=np.int64),
                 "target_eos_mask": np.array([448, 49, 50, 51], dtype=np.int64)},
                {"source_sos_ids": np.array([51, 2, 3, 4, 5], dtype=np.int64),
                 "source_sos_mask": np.array([56, 7, 8, 9, 10, 11, 12], dtype=np.int64),
                 "source_eos_ids": np.array([513, 14, 15, 16, 17, 18], dtype=np.int64),
                 "source_eos_mask": np.array([519, 20, 21, 22, 23, 24, 25, 26, 27], dtype=np.int64),
                 "target_sos_ids": np.array([528, 29, 30, 31, 32], dtype=np.int64),
                 "target_sos_mask": np.array([533, 34, 35, 36, 37, 38], dtype=np.int64),
                 "target_eos_ids": np.array([539, 40, 41, 42, 43, 44, 45, 46, 47], dtype=np.int64),
                 "target_eos_mask": np.array([548, 49, 50, 51], dtype=np.int64)}
                ]
        writer = FileWriter(mindrecord_file_name)
        schema = {"source_sos_ids": {"type": "int64", "shape": [-1]},
                  "source_sos_mask": {"type": "int64", "shape": [-1]},
                  "source_eos_ids": {"type": "int64", "shape": [-1]},
                  "source_eos_mask": {"type": "int64", "shape": [-1]},
                  "target_sos_ids": {"type": "int64", "shape": [-1]},
                  "target_sos_mask": {"type": "int64", "shape": [-1]},
                  "target_eos_ids": {"type": "int64", "shape": [-1]},
                  "target_eos_mask": {"type": "int64", "shape": [-1]}}
        writer.add_schema(schema, "data is so cool")
        writer.write_raw_data(data)
        writer.commit()

        # change data value to list - do none
        data_value_to_list = []
        for item in data:
            new_data = {}
            new_data['source_sos_ids'] = item["source_sos_ids"]
            new_data['source_sos_mask'] = item["source_sos_mask"]
            new_data['source_eos_ids'] = item["source_eos_ids"]
            new_data['source_eos_mask'] = item["source_eos_mask"]
            new_data['target_sos_ids'] = item["target_sos_ids"]
            new_data['target_sos_mask'] = item["target_sos_mask"]
            new_data['target_eos_ids'] = item["target_eos_ids"]
            new_data['target_eos_mask'] = item["target_eos_mask"]
            data_value_to_list.append(new_data)

        num_readers = 2
        data_set = ds.MindDataset(dataset_files=mindrecord_file_name,
                                  num_parallel_workers=num_readers,
                                  shuffle=False)
        assert data_set.get_dataset_size() == 6
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            assert len(item) == 8
            for field in item:
                if isinstance(item[field], np.ndarray):
                    assert (item[field] ==
                            data_value_to_list[num_iter][field]).all()
                else:
                    assert item[field] == data_value_to_list[num_iter][field]
            num_iter += 1
        assert num_iter == 6

        num_readers = 2
        data_set = ds.MindDataset(dataset_files=mindrecord_file_name,
                                  columns_list=["source_eos_ids", "source_eos_mask",
                                                "target_sos_ids", "target_sos_mask",
                                                "target_eos_ids", "target_eos_mask"],
                                  num_parallel_workers=num_readers,
                                  shuffle=False)
        assert data_set.get_dataset_size() == 6
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            assert len(item) == 6
            for field in item:
                if isinstance(item[field], np.ndarray):
                    assert (item[field] ==
                            data_value_to_list[num_iter][field]).all()
                else:
                    assert item[field] == data_value_to_list[num_iter][field]
            num_iter += 1
        assert num_iter == 6

        num_readers = 2
        data_set = ds.MindDataset(dataset_files=mindrecord_file_name,
                                  columns_list=["source_sos_ids",
                                                "target_sos_ids",
                                                "target_eos_mask"],
                                  num_parallel_workers=num_readers,
                                  shuffle=False)
        assert data_set.get_dataset_size() == 6
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            assert len(item) == 3
            for field in item:
                if isinstance(item[field], np.ndarray):
                    assert (item[field] ==
                            data_value_to_list[num_iter][field]).all()
                else:
                    assert item[field] == data_value_to_list[num_iter][field]
            num_iter += 1
        assert num_iter == 6

        num_readers = 2
        data_set = ds.MindDataset(dataset_files=mindrecord_file_name,
                                  columns_list=["target_eos_mask",
                                                "source_eos_mask",
                                                "source_sos_mask"],
                                  num_parallel_workers=num_readers,
                                  shuffle=False)
        assert data_set.get_dataset_size() == 6
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            assert len(item) == 3
            for field in item:
                if isinstance(item[field], np.ndarray):
                    assert (item[field] ==
                            data_value_to_list[num_iter][field]).all()
                else:
                    assert item[field] == data_value_to_list[num_iter][field]
            num_iter += 1
        assert num_iter == 6

        num_readers = 2
        data_set = ds.MindDataset(dataset_files=mindrecord_file_name,
                                  columns_list=["target_eos_ids"],
                                  num_parallel_workers=num_readers,
                                  shuffle=False)
        assert data_set.get_dataset_size() == 6
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            assert len(item) == 1
            for field in item:
                if isinstance(item[field], np.ndarray):
                    assert (item[field] ==
                            data_value_to_list[num_iter][field]).all()
                else:
                    assert item[field] == data_value_to_list[num_iter][field]
            num_iter += 1
        assert num_iter == 6

        num_readers = 1
        data_set = ds.MindDataset(dataset_files=mindrecord_file_name,
                                  columns_list=["target_eos_mask", "target_eos_ids",
                                                "target_sos_mask", "target_sos_ids",
                                                "source_eos_mask", "source_eos_ids",
                                                "source_sos_mask", "source_sos_ids"],
                                  num_parallel_workers=num_readers,
                                  shuffle=False)
        assert data_set.get_dataset_size() == 6
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            assert len(item) == 8
            for field in item:
                if isinstance(item[field], np.ndarray):
                    assert (item[field] ==
                            data_value_to_list[num_iter][field]).all()
                else:
                    assert item[field] == data_value_to_list[num_iter][field]
            num_iter += 1
        assert num_iter == 6
    except Exception as error:
        os.remove("{}".format(mindrecord_file_name))
        os.remove("{}.db".format(mindrecord_file_name))
        raise error
    else:
        os.remove("{}".format(mindrecord_file_name))
        os.remove("{}.db".format(mindrecord_file_name))


def test_numpy_generic():
    """
    Feature: MindDataset
    Description: Test write numpy generic data types using FileWriter and read them by MindDataset
    Expectation: Output is equal to the expected output
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    paths = ["{}{}".format(file_name, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    try:
        for x in paths:
            if os.path.exists("{}".format(x)):
                os.remove("{}".format(x))
            if os.path.exists("{}.db".format(x)):
                os.remove("{}.db".format(x))
        writer = FileWriter(file_name, FILES_NUM)
        cv_schema_json = {"label1": {"type": "int32"}, "label2": {"type": "int64"},
                          "label3": {"type": "float32"}, "label4": {"type": "float64"}}
        data = []
        for idx in range(10):
            row = {}
            row['label1'] = np.int32(idx)
            row['label2'] = np.int64(idx * 10)
            row['label3'] = np.float32(idx + 0.12345)
            row['label4'] = np.float64(idx + 0.12345789)
            data.append(row)
        writer.add_schema(cv_schema_json, "img_schema")
        writer.write_raw_data(data)
        writer.commit()

        num_readers = 4
        data_set = ds.MindDataset(file_name + "0", None, num_readers, shuffle=False)
        assert data_set.get_dataset_size() == 10
        idx = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            assert item['label1'] == item['label1']
            assert item['label2'] == item['label2']
            assert item['label3'] == item['label3']
            assert item['label4'] == item['label4']
            idx += 1
        assert idx == 10
    except Exception as error:
        for x in paths:
            os.remove("{}".format(x))
            os.remove("{}.db".format(x))
        raise error
    else:
        for x in paths:
            os.remove("{}".format(x))
            os.remove("{}.db".format(x))


def test_write_with_float32_float64_float32_array_float64_array_and_MindDataset():
    """
    Feature: MindDataset
    Description: Test write float32, float64, array of float32, and array of float64 using
        FileWriter and read them by MindDataset
    Expectation: Output is equal to the expected output
    """
    mindrecord_file_name = "test_write_with_float32_float64_float32_array_float64_array_and_MindDataset.mindrecord"
    try:
        data = [{"float32_array": np.array([1.2, 2.78, 3.1234, 4.9871, 5.12341], dtype=np.float32),
                 "float64_array": np.array([48.1234556789, 49.3251241431, 50.13514312414, 51.8971298471,
                                            123414314.2141243, 87.1212122], dtype=np.float64),
                 "float32": 3456.12345,
                 "float64": 1987654321.123456785,
                 "int32_array": np.array([1, 2, 3, 4, 5], dtype=np.int32),
                 "int64_array": np.array([48, 49, 50, 51, 123414314, 87], dtype=np.int64),
                 "int32": 3456,
                 "int64": 947654321123},
                {"float32_array": np.array([1.2, 2.78, 4.1234, 4.9871, 5.12341], dtype=np.float32),
                 "float64_array": np.array([48.1234556789, 49.3251241431, 60.13514312414, 51.8971298471,
                                            123414314.2141243, 87.1212122], dtype=np.float64),
                 "float32": 3456.12445,
                 "float64": 1987654321.123456786,
                 "int32_array": np.array([11, 21, 31, 41, 51], dtype=np.int32),
                 "int64_array": np.array([481, 491, 501, 511, 1234143141, 871], dtype=np.int64),
                 "int32": 3466,
                 "int64": 957654321123},
                {"float32_array": np.array([1.2, 2.78, 5.1234, 4.9871, 5.12341], dtype=np.float32),
                 "float64_array": np.array([48.1234556789, 49.3251241431, 70.13514312414, 51.8971298471,
                                            123414314.2141243, 87.1212122], dtype=np.float64),
                 "float32": 3456.12545,
                 "float64": 1987654321.123456787,
                 "int32_array": np.array([12, 22, 32, 42, 52], dtype=np.int32),
                 "int64_array": np.array([482, 492, 502, 512, 1234143142, 872], dtype=np.int64),
                 "int32": 3476,
                 "int64": 967654321123},
                {"float32_array": np.array([1.2, 2.78, 6.1234, 4.9871, 5.12341], dtype=np.float32),
                 "float64_array": np.array([48.1234556789, 49.3251241431, 80.13514312414, 51.8971298471,
                                            123414314.2141243, 87.1212122], dtype=np.float64),
                 "float32": 3456.12645,
                 "float64": 1987654321.123456788,
                 "int32_array": np.array([13, 23, 33, 43, 53], dtype=np.int32),
                 "int64_array": np.array([483, 493, 503, 513, 1234143143, 873], dtype=np.int64),
                 "int32": 3486,
                 "int64": 977654321123},
                {"float32_array": np.array([1.2, 2.78, 7.1234, 4.9871, 5.12341], dtype=np.float32),
                 "float64_array": np.array([48.1234556789, 49.3251241431, 90.13514312414, 51.8971298471,
                                            123414314.2141243, 87.1212122], dtype=np.float64),
                 "float32": 3456.12745,
                 "float64": 1987654321.123456789,
                 "int32_array": np.array([14, 24, 34, 44, 54], dtype=np.int32),
                 "int64_array": np.array([484, 494, 504, 514, 1234143144, 874], dtype=np.int64),
                 "int32": 3496,
                 "int64": 987654321123},
                ]
        writer = FileWriter(mindrecord_file_name)
        schema = {"float32_array": {"type": "float32", "shape": [-1]},
                  "float64_array": {"type": "float64", "shape": [-1]},
                  "float32": {"type": "float32"},
                  "float64": {"type": "float64"},
                  "int32_array": {"type": "int32", "shape": [-1]},
                  "int64_array": {"type": "int64", "shape": [-1]},
                  "int32": {"type": "int32"},
                  "int64": {"type": "int64"}}
        writer.add_schema(schema, "data is so cool")
        writer.write_raw_data(data)
        writer.commit()

        # change data value to list - do none
        data_value_to_list = []
        for item in data:
            new_data = {}
            new_data['float32_array'] = item["float32_array"]
            new_data['float64_array'] = item["float64_array"]
            new_data['float32'] = item["float32"]
            new_data['float64'] = item["float64"]
            new_data['int32_array'] = item["int32_array"]
            new_data['int64_array'] = item["int64_array"]
            new_data['int32'] = item["int32"]
            new_data['int64'] = item["int64"]
            data_value_to_list.append(new_data)

        num_readers = 2
        data_set = ds.MindDataset(dataset_files=mindrecord_file_name,
                                  num_parallel_workers=num_readers,
                                  shuffle=False)
        assert data_set.get_dataset_size() == 5
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            assert len(item) == 8
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
        assert num_iter == 5

        num_readers = 2
        data_set = ds.MindDataset(dataset_files=mindrecord_file_name,
                                  columns_list=["float32", "int32"],
                                  num_parallel_workers=num_readers,
                                  shuffle=False)
        assert data_set.get_dataset_size() == 5
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            assert len(item) == 2
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
        assert num_iter == 5

        num_readers = 2
        data_set = ds.MindDataset(dataset_files=mindrecord_file_name,
                                  columns_list=["float64", "int64"],
                                  num_parallel_workers=num_readers,
                                  shuffle=False)
        assert data_set.get_dataset_size() == 5
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            assert len(item) == 2
            for field in item:
                if isinstance(item[field], np.ndarray):
                    if item[field].dtype == np.float32:
                        assert (item[field] ==
                                np.array(data_value_to_list[num_iter][field], np.float32)).all()
                    elif item[field].dtype == np.float64:
                        assert math.isclose(item[field],
                                            np.array(data_value_to_list[num_iter][field], np.float64),
                                            rel_tol=1e-14)
                    else:
                        assert (item[field] ==
                                data_value_to_list[num_iter][field]).all()
                else:
                    assert item[field] == data_value_to_list[num_iter][field]
            num_iter += 1
        assert num_iter == 5
    except Exception as error:
        os.remove("{}".format(mindrecord_file_name))
        os.remove("{}.db".format(mindrecord_file_name))
        raise error
    else:
        os.remove("{}".format(mindrecord_file_name))
        os.remove("{}.db".format(mindrecord_file_name))

@pytest.fixture
def create_multi_mindrecord_files():
    """files: {0.mindrecord : 10, 1.mindrecord : 14, 2.mindrecord : 8, 3.mindrecord : 20}"""
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    files = [file_name + str(idx) for idx in range(4)]
    items = [10, 14, 8, 20]
    file_items = {files[0]: items[0], files[1]: items[1], files[2]: items[2], files[3]: items[3]}

    try:
        index = 0
        for key in file_items:
            if os.path.exists(key):
                os.remove("{}".format(key))
                os.remove("{}.db".format(key))

            value = file_items[key]
            data_list = []
            for i in range(value):
                data = {}
                data['id'] = i + index
                data_list.append(data)
            index += value

            writer = FileWriter(key)
            schema = {"id": {"type": "int32"}}
            writer.add_schema(schema, "data is so cool")
            writer.write_raw_data(data_list)
            writer.commit()
        yield "yield_create_multi_mindrecord_files"
    except Exception as error:
        for filename in file_items:
            if os.path.exists(filename):
                os.remove("{}".format(filename))
                os.remove("{}.db".format(filename))
        raise error
    else:
        for filename in file_items:
            if os.path.exists(filename):
                os.remove("{}".format(filename))
                os.remove("{}.db".format(filename))


def test_shuffle_with_global_infile_files(create_multi_mindrecord_files):
    """
    Feature: MindDataset
    Description: Test without and with shuffle args for MindDataset
    Expectation: Output is equal to the expected output
    """
    original_seed = config_get_set_seed(1)
    datas_all = []
    index = 0
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    files = [file_name + str(idx) for idx in range(4)]
    items = [10, 14, 8, 20]
    file_items = {files[0]: items[0], files[1]: items[1], files[2]: items[2], files[3]: items[3]}
    for filename in file_items:
        value = file_items[filename]
        data_list = []
        for i in range(value):
            data = {}
            data['id'] = np.array(i + index, dtype=np.int32)
            data_list.append(data)
        index += value
        datas_all.append(data_list)

    # no shuffle parameter
    num_readers = 2
    data_set = ds.MindDataset(dataset_files=files,
                              num_parallel_workers=num_readers)
    assert data_set.get_dataset_size() == 52
    num_iter = 0
    datas_all_minddataset = []
    data_list = []
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 1
        data_list.append(item)
        if num_iter == 9:
            datas_all_minddataset.append(data_list)
            data_list = []
        elif num_iter == 23:
            datas_all_minddataset.append(data_list)
            data_list = []
        elif num_iter == 31:
            datas_all_minddataset.append(data_list)
            data_list = []
        elif num_iter == 51:
            datas_all_minddataset.append(data_list)
            data_list = []
        num_iter += 1
    assert data_set.get_dataset_size() == 52

    assert len(datas_all) == len(datas_all_minddataset)
    for i, _ in enumerate(datas_all):
        assert len(datas_all[i]) == len(datas_all_minddataset[i])
        assert datas_all[i] != datas_all_minddataset[i]

    # shuffle=False
    num_readers = 2
    data_set = ds.MindDataset(dataset_files=files,
                              num_parallel_workers=num_readers,
                              shuffle=False)
    assert data_set.get_dataset_size() == 52
    num_iter = 0
    datas_all_minddataset = []
    data_list = []
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 1
        data_list.append(item)
        if num_iter == 9:
            datas_all_minddataset.append(data_list)
            data_list = []
        elif num_iter == 23:
            datas_all_minddataset.append(data_list)
            data_list = []
        elif num_iter == 31:
            datas_all_minddataset.append(data_list)
            data_list = []
        elif num_iter == 51:
            datas_all_minddataset.append(data_list)
            data_list = []
        num_iter += 1
    assert data_set.get_dataset_size() == 52

    assert len(datas_all) == len(datas_all_minddataset)
    for i, _ in enumerate(datas_all):
        assert len(datas_all[i]) == len(datas_all_minddataset[i])
        assert datas_all[i] == datas_all_minddataset[i]

    # shuffle=True
    num_readers = 2
    data_set = ds.MindDataset(dataset_files=files,
                              num_parallel_workers=num_readers,
                              shuffle=True)
    assert data_set.get_dataset_size() == 52
    num_iter = 0
    datas_all_minddataset = []
    data_list = []
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 1
        data_list.append(item)
        if num_iter == 9:
            datas_all_minddataset.append(data_list)
            data_list = []
        elif num_iter == 23:
            datas_all_minddataset.append(data_list)
            data_list = []
        elif num_iter == 31:
            datas_all_minddataset.append(data_list)
            data_list = []
        elif num_iter == 51:
            datas_all_minddataset.append(data_list)
            data_list = []
        num_iter += 1
    assert data_set.get_dataset_size() == 52

    assert len(datas_all) == len(datas_all_minddataset)
    for i, _ in enumerate(datas_all):
        assert len(datas_all[i]) == len(datas_all_minddataset[i])
        assert datas_all[i] != datas_all_minddataset[i]

    # shuffle=Shuffle.GLOBAL
    num_readers = 2
    data_set = ds.MindDataset(dataset_files=files,
                              num_parallel_workers=num_readers,
                              shuffle=ds.Shuffle.GLOBAL)
    assert data_set.get_dataset_size() == 52
    num_iter = 0
    datas_all_minddataset = []
    data_list = []
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 1
        data_list.append(item)
        if num_iter == 9:
            datas_all_minddataset.append(data_list)
            data_list = []
        elif num_iter == 23:
            datas_all_minddataset.append(data_list)
            data_list = []
        elif num_iter == 31:
            datas_all_minddataset.append(data_list)
            data_list = []
        elif num_iter == 51:
            datas_all_minddataset.append(data_list)
            data_list = []
        num_iter += 1
    assert data_set.get_dataset_size() == 52

    assert len(datas_all) == len(datas_all_minddataset)
    for i, _ in enumerate(datas_all):
        assert len(datas_all[i]) == len(datas_all_minddataset[i])
        assert datas_all[i] != datas_all_minddataset[i]

    # shuffle=Shuffle.INFILE
    num_readers = 2
    data_set = ds.MindDataset(dataset_files=files,
                              num_parallel_workers=num_readers,
                              shuffle=ds.Shuffle.INFILE)
    assert data_set.get_dataset_size() == 52
    num_iter = 0
    datas_all_minddataset = []
    data_list = []
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 1
        data_list.append(item)
        if num_iter == 9:
            datas_all_minddataset.append(data_list)
            data_list = []
        elif num_iter == 23:
            datas_all_minddataset.append(data_list)
            data_list = []
        elif num_iter == 31:
            datas_all_minddataset.append(data_list)
            data_list = []
        elif num_iter == 51:
            datas_all_minddataset.append(data_list)
            data_list = []
        num_iter += 1
    assert data_set.get_dataset_size() == 52

    def sort_list_with_dict(dict_in_list):
        keys = []
        for item in dict_in_list:
            for key in item:
                keys.append(int(item[key]))
        keys.sort()
        data_list = []
        for item in keys:
            data = {}
            data['id'] = np.array(item, dtype=np.int32)
            data_list.append(data)
        return data_list

    assert len(datas_all) == len(datas_all_minddataset)
    for i, _ in enumerate(datas_all):
        assert len(datas_all[i]) == len(datas_all_minddataset[i])
        assert datas_all[i] != datas_all_minddataset[i]
        # order the datas_all_minddataset
        new_datas_all_minddataset = sort_list_with_dict(datas_all_minddataset[i])
        assert datas_all[i] == new_datas_all_minddataset

    # shuffle=Shuffle.FILES
    num_readers = 2
    data_set = ds.MindDataset(dataset_files=files,
                              num_parallel_workers=num_readers,
                              shuffle=ds.Shuffle.FILES)
    assert data_set.get_dataset_size() == 52

    num_iter = 0
    data_list = []

    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 1
        data_list.append(item)
        num_iter += 1
    assert data_set.get_dataset_size() == 52

    current_shard_size = 0
    current_shard_index = 0
    shard_count = 0
    datas_index = 0
    origin_index = [i for i in range(len(items))]
    current_index = []
    while shard_count < len(items):
        if data_list[datas_index]['id'] < 10:
            current_shard_index = 0
        elif data_list[datas_index]['id'] < 24:
            current_shard_index = 1
        elif data_list[datas_index]['id'] < 32:
            current_shard_index = 2
        elif data_list[datas_index]['id'] < 52:
            current_shard_index = 3
        else:
            raise ValueError("Index out of range")
        current_shard_size = items[current_shard_index]

        tmp_datas = data_list[datas_index:datas_index + current_shard_size]
        current_index.append(current_shard_index)
        assert len(datas_all[current_shard_index]) == len(tmp_datas)
        assert datas_all[current_shard_index] == tmp_datas

        datas_index += current_shard_size
        shard_count += 1
    assert origin_index != current_index

    ds.config.set_seed(original_seed)


def test_distributed_shuffle_with_global_infile_files(create_multi_mindrecord_files):
    """
    Feature: MindDataset
    Description: Test distributed MindDataset (with num_shards and shard_id) without and with shuffle args
    Expectation: Output is equal to the expected output
    """
    original_seed = config_get_set_seed(1)
    datas_all = []
    datas_all_samples = []
    index = 0
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    files = [file_name + str(idx) for idx in range(4)]
    items = [10, 14, 8, 20]
    file_items = {files[0]: items[0], files[1]: items[1], files[2]: items[2], files[3]: items[3]}
    for filename in file_items:
        value = file_items[filename]
        data_list = []
        for i in range(value):
            data = {}
            data['id'] = np.array(i + index, dtype=np.int32)
            data_list.append(data)
            datas_all_samples.append(data)
        index += value
        datas_all.append(data_list)

    # no shuffle parameter
    num_readers = 2
    data_set = ds.MindDataset(dataset_files=files,
                              num_parallel_workers=num_readers,
                              num_shards=4,
                              shard_id=3)
    assert data_set.get_dataset_size() == 13
    num_iter = 0
    data_list = []
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 1
        data_list.append(item)
        num_iter += 1
    assert num_iter == 13
    assert data_list != datas_all_samples[3*13:]

    # shuffle=False
    num_readers = 2
    data_set = ds.MindDataset(dataset_files=files,
                              num_parallel_workers=num_readers,
                              shuffle=False,
                              num_shards=4,
                              shard_id=2)
    assert data_set.get_dataset_size() == 13
    num_iter = 0
    data_list = []
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 1
        data_list.append(item)
        num_iter += 1
    assert num_iter == 13
    assert data_list == datas_all_samples[2*13:3*13]

    # shuffle=True
    num_readers = 2
    data_set = ds.MindDataset(dataset_files=files,
                              num_parallel_workers=num_readers,
                              shuffle=True,
                              num_shards=4,
                              shard_id=1)
    assert data_set.get_dataset_size() == 13
    num_iter = 0
    data_list = []
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 1
        data_list.append(item)
        num_iter += 1
    assert num_iter == 13
    assert data_list != datas_all_samples[1*13:2*13]

    # shuffle=Shuffle.GLOBAL
    num_readers = 2
    data_set = ds.MindDataset(dataset_files=files,
                              num_parallel_workers=num_readers,
                              shuffle=ds.Shuffle.GLOBAL,
                              num_shards=4,
                              shard_id=0)
    assert data_set.get_dataset_size() == 13
    num_iter = 0
    data_list = []
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 1
        data_list.append(item)
        num_iter += 1
    assert num_iter == 13
    assert data_list != datas_all_samples[0:1*13]

    # shuffle=Shuffle.INFILE
    output_datas = []
    for shard_id in range(4):
        num_readers = 2
        data_set = ds.MindDataset(dataset_files=files,
                                  num_parallel_workers=num_readers,
                                  shuffle=ds.Shuffle.INFILE,
                                  num_shards=4,
                                  shard_id=shard_id)
        assert data_set.get_dataset_size() == 13
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            assert len(item) == 1
            output_datas.append(item)
            num_iter += 1
        assert num_iter == 13

    num_iter = 0
    datas_all_minddataset = []
    data_list = []
    for item in output_datas:
        assert len(item) == 1
        data_list.append(item)
        if num_iter == 9:
            datas_all_minddataset.append(data_list)
            data_list = []
        elif num_iter == 23:
            datas_all_minddataset.append(data_list)
            data_list = []
        elif num_iter == 31:
            datas_all_minddataset.append(data_list)
            data_list = []
        elif num_iter == 51:
            datas_all_minddataset.append(data_list)
            data_list = []
        num_iter += 1
    assert num_iter == 52

    def sort_list_with_dict(dict_in_list):
        keys = []
        for item in dict_in_list:
            for key in item:
                keys.append(int(item[key]))
        keys.sort()
        data_list = []
        for item in keys:
            data = {}
            data['id'] = np.array(item, dtype=np.int32)
            data_list.append(data)
        return data_list

    assert len(datas_all) == len(datas_all_minddataset)
    for i, _ in enumerate(datas_all):
        assert len(datas_all[i]) == len(datas_all_minddataset[i])
        assert datas_all[i] != datas_all_minddataset[i]
        # order the datas_all_minddataset
        new_datas_all_minddataset = sort_list_with_dict(datas_all_minddataset[i])
        assert datas_all[i] == new_datas_all_minddataset

    # shuffle=Shuffle.Files
    data_list = []
    for shard_id in range(4):
        num_readers = 2
        data_set = ds.MindDataset(dataset_files=files,
                                  num_parallel_workers=num_readers,
                                  shuffle=ds.Shuffle.FILES,
                                  num_shards=4,
                                  shard_id=shard_id)
        assert data_set.get_dataset_size() == 13
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            assert len(item) == 1
            data_list.append(item)
            num_iter += 1
        assert num_iter == 13
    assert len(data_list) == 52

    current_shard_size = 0
    current_shard_index = 0
    shard_count = 0
    datas_index = 0
    origin_index = [i for i in range(len(items))]
    current_index = []
    while shard_count < len(items):
        if data_list[datas_index]['id'] < 10:
            current_shard_index = 0
        elif data_list[datas_index]['id'] < 24:
            current_shard_index = 1
        elif data_list[datas_index]['id'] < 32:
            current_shard_index = 2
        elif data_list[datas_index]['id'] < 52:
            current_shard_index = 3
        else:
            raise ValueError("Index out of range")
        current_shard_size = items[current_shard_index]

        tmp_datas = data_list[datas_index:datas_index + current_shard_size]
        current_index.append(current_shard_index)
        assert len(datas_all[current_shard_index]) == len(tmp_datas)
        assert datas_all[current_shard_index] == tmp_datas

        datas_index += current_shard_size
        shard_count += 1
    assert origin_index != current_index

    ds.config.set_seed(original_seed)


def test_distributed_shuffle_with_multi_epochs(create_multi_mindrecord_files):
    """
    Feature: MindDataset
    Description: Test distributed MindDataset (with num_shards and shard_id)
        without and with shuffle args under multiple epochs
    Expectation: Output is equal to the expected output
    """
    original_seed = config_get_set_seed(1)
    datas_all = []
    datas_all_samples = []
    index = 0
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    files = [file_name + str(idx) for idx in range(4)]
    items = [10, 14, 8, 20]
    file_items = {files[0]: items[0], files[1]: items[1], files[2]: items[2], files[3]: items[3]}
    for filename in file_items:
        value = file_items[filename]
        data_list = []
        for i in range(value):
            data = {}
            data['id'] = np.array(i + index, dtype=np.int32)
            data_list.append(data)
            datas_all_samples.append(data)
        index += value
        datas_all.append(data_list)

    epoch_size = 3

    # no shuffle parameter
    for shard_id in range(4):
        num_readers = 2
        data_set = ds.MindDataset(dataset_files=files,
                                  num_parallel_workers=num_readers,
                                  num_shards=4,
                                  shard_id=shard_id)
        assert data_set.get_dataset_size() == 13
        data_list = []
        dataset_iter = data_set.create_dict_iterator(num_epochs=epoch_size, output_numpy=True)
        for epoch in range(epoch_size):  # 3 epoch
            num_iter = 0
            new_datas = []
            for item in dataset_iter:
                assert len(item) == 1
                new_datas.append(item)
                num_iter += 1
            assert num_iter == 13
            assert new_datas != datas_all_samples[shard_id*13:(shard_id+1)*13]
            assert data_list != new_datas
            data_list = new_datas

    # shuffle=False
    for shard_id in range(4):
        num_readers = 2
        data_set = ds.MindDataset(dataset_files=files,
                                  num_parallel_workers=num_readers,
                                  shuffle=False,
                                  num_shards=4,
                                  shard_id=shard_id)
        assert data_set.get_dataset_size() == 13
        data_list = []
        dataset_iter = data_set.create_dict_iterator(num_epochs=epoch_size, output_numpy=True)
        for epoch in range(epoch_size):  # 3 epoch
            num_iter = 0
            new_datas = []
            for item in dataset_iter:
                assert len(item) == 1
                new_datas.append(item)
                num_iter += 1
            assert num_iter == 13
            assert new_datas == datas_all_samples[shard_id*13:(shard_id+1)*13]

    # shuffle=True
    for shard_id in range(4):
        num_readers = 2
        data_set = ds.MindDataset(dataset_files=files,
                                  num_parallel_workers=num_readers,
                                  shuffle=True,
                                  num_shards=4,
                                  shard_id=shard_id)
        assert data_set.get_dataset_size() == 13
        data_list = []
        dataset_iter = data_set.create_dict_iterator(num_epochs=epoch_size, output_numpy=True)
        for epoch in range(epoch_size):  # 3 epoch
            num_iter = 0
            new_datas = []
            for item in dataset_iter:
                assert len(item) == 1
                new_datas.append(item)
                num_iter += 1
            assert num_iter == 13
            assert new_datas != datas_all_samples[shard_id*13:(shard_id+1)*13]
            assert data_list != new_datas
            data_list = new_datas

    # shuffle=Shuffle.GLOBAL
    for shard_id in range(4):
        num_readers = 2
        data_set = ds.MindDataset(dataset_files=files,
                                  num_parallel_workers=num_readers,
                                  shuffle=ds.Shuffle.GLOBAL,
                                  num_shards=4,
                                  shard_id=shard_id)
        assert data_set.get_dataset_size() == 13
        data_list = []
        dataset_iter = data_set.create_dict_iterator(num_epochs=epoch_size, output_numpy=True)
        for epoch in range(epoch_size):  # 3 epoch
            num_iter = 0
            new_datas = []
            for item in dataset_iter:
                assert len(item) == 1
                new_datas.append(item)
                num_iter += 1
            assert num_iter == 13
            assert new_datas != datas_all_samples[shard_id*13:(shard_id+1)*13]
            assert data_list != new_datas
            data_list = new_datas

    # shuffle=Shuffle.INFILE
    for shard_id in range(4):
        num_readers = 2
        data_set = ds.MindDataset(dataset_files=files,
                                  num_parallel_workers=num_readers,
                                  shuffle=ds.Shuffle.INFILE,
                                  num_shards=4,
                                  shard_id=shard_id)
        assert data_set.get_dataset_size() == 13
        data_list = []
        dataset_iter = data_set.create_dict_iterator(num_epochs=epoch_size, output_numpy=True)
        for epoch in range(epoch_size):  # 3 epoch
            num_iter = 0
            new_datas = []
            for item in dataset_iter:
                assert len(item) == 1
                new_datas.append(item)
                num_iter += 1
            assert num_iter == 13
            assert new_datas != datas_all_samples[shard_id*13:(shard_id+1)*13]
            assert data_list != new_datas
            data_list = new_datas

    # shuffle=Shuffle.FILES
    datas_epoch1 = []
    datas_epoch2 = []
    datas_epoch3 = []
    for shard_id in range(4):
        num_readers = 2
        data_set = ds.MindDataset(dataset_files=files,
                                  num_parallel_workers=num_readers,
                                  shuffle=ds.Shuffle.FILES,
                                  num_shards=4,
                                  shard_id=shard_id)
        assert data_set.get_dataset_size() == 13
        dataset_iter = data_set.create_dict_iterator(num_epochs=epoch_size, output_numpy=True)
        for epoch in range(epoch_size):  # 3 epoch
            num_iter = 0
            for item in dataset_iter:
                assert len(item) == 1
                if epoch == 0:
                    datas_epoch1.append(item)
                elif epoch == 1:
                    datas_epoch2.append(item)
                elif epoch == 2:
                    datas_epoch3.append(item)
                num_iter += 1
            assert num_iter == 13
    assert datas_epoch1 not in (datas_epoch2, datas_epoch3)
    assert datas_epoch2 not in (datas_epoch1, datas_epoch3)
    assert datas_epoch3 not in (datas_epoch2, datas_epoch1)

    ds.config.set_seed(original_seed)


def test_field_is_null_numpy():
    """
    Feature: MindDataset
    Description: Test MindDataset when field array_d is null
    Expectation: Output is equal to the expected output
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    paths = ["{}{}".format(file_name, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    for x in paths:
        if os.path.exists("{}".format(x)):
            os.remove("{}".format(x))
        if os.path.exists("{}.db".format(x)):
            os.remove("{}.db".format(x))

    writer = FileWriter(file_name, FILES_NUM)
    data = []
    # field array_d is null
    for row_id in range(16):
        data.append({
            "label": row_id,
            "array_a": np.reshape(np.array([0, 1, -1, 127, -128, 128, -129,
                                            255, 256, -32768, 32767, -32769, 32768, -2147483648,
                                            2147483647], dtype=np.int32), [-1]),
            "array_b": np.reshape(np.array([0, 1, -1, 127, -128, 128, -129, 255,
                                            256, -32768, 32767, -32769, 32768,
                                            -2147483648, 2147483647, -2147483649, 2147483649,
                                            -922337036854775808, 9223372036854775807]), [1, -1]),
            "array_d": np.array([], dtype=np.int64)
        })
    nlp_schema_json = {"label": {"type": "int32"},
                       "array_a": {"type": "int32",
                                   "shape": [-1]},
                       "array_b": {"type": "int64",
                                   "shape": [1, -1]},
                       "array_d": {"type": "int64",
                                   "shape": [-1]}
                       }
    writer.set_header_size(1 << 14)
    writer.set_page_size(1 << 15)
    writer.add_schema(nlp_schema_json, "nlp_schema")
    writer.write_raw_data(data)
    writer.commit()

    data_set = ds.MindDataset(dataset_files=file_name + "0",
                              columns_list=["label", "array_a", "array_b", "array_d"],
                              num_parallel_workers=2,
                              shuffle=False)
    assert data_set.get_dataset_size() == 16
    assert data_set.output_shapes() == [[], [15], [1, 19], []]
    assert data_set.output_types()[0] == np.int32
    assert data_set.output_types()[1] == np.int32
    assert data_set.output_types()[2] == np.int64
    assert data_set.output_types()[3] == np.int64

    count = 0
    for item in data_set.create_dict_iterator(output_numpy=True):
        assert item["label"] == np.array(count)
        assert (item["array_a"] == np.array([0, 1, -1, 127, -128, 128, -129, 255, 256, -32768,
                                             32767, -32769, 32768, -2147483648, 2147483647])).all()
        assert (item["array_b"] == np.array([0, 1, -1, 127, -128, 128, -129, 255, 256, -32768,
                                             32767, -32769, 32768, -2147483648, 2147483647,
                                             -2147483649, 2147483649, -922337036854775808,
                                             9223372036854775807])).all()
        assert (item["array_d"] == np.array([])).all()
        count += 1

    for x in paths:
        os.remove("{}".format(x))
        os.remove("{}.db".format(x))


def test_for_loop_dataset_iterator(add_and_remove_nlp_compress_file):
    """
    Feature: MindDataset
    Description: Test for loop for iterator based on MindDataset
    Expectation: Output is equal to the expected output
    """
    data = []
    for row_id in range(16):
        data.append({
            "label": row_id,
            "array_a": np.reshape(np.array([0, 1, -1, 127, -128, 128, -129,
                                            255, 256, -32768, 32767, -32769, 32768, -2147483648,
                                            2147483647], dtype=np.int32), [-1]),
            "array_b": np.reshape(np.array([0, 1, -1, 127, -128, 128, -129, 255,
                                            256, -32768, 32767, -32769, 32768,
                                            -2147483648, 2147483647, -2147483649, 2147483649,
                                            -922337036854775808, 9223372036854775807]), [1, -1]),
            "array_c": str.encode("nlp data"),
            "array_d": np.reshape(np.array([[-10, -127], [10, 127]]), [2, -1])
        })
    num_readers = 1
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset(
        file_name + "0", None, num_readers, shuffle=False)
    assert data_set.get_dataset_size() == 16

    # create_dict_iterator in for loop
    for _ in range(10):
        num_iter = 0
        for x, item in zip(data, data_set.create_dict_iterator(num_epochs=1, output_numpy=True)):
            assert (item["array_a"] == x["array_a"]).all()
            assert (item["array_b"] == x["array_b"]).all()
            assert item["array_c"].tobytes() == x["array_c"]
            assert (item["array_d"] == x["array_d"]).all()
            assert item["label"] == x["label"]
            num_iter += 1
        assert num_iter == 16

    # create_dict_iterator beyond for loop
    dataset_iter = data_set.create_dict_iterator(num_epochs=10, output_numpy=True)
    new_data = data * 10
    for _ in range(10):
        num_iter = 0
        for x, item in zip(new_data, dataset_iter):
            assert (item["array_a"] == x["array_a"]).all()
            assert (item["array_b"] == x["array_b"]).all()
            assert item["array_c"].tobytes() == x["array_c"]
            assert (item["array_d"] == x["array_d"]).all()
            assert item["label"] == x["label"]
            num_iter += 1
        assert num_iter == 16

    # create mulit iter by user
    dataset_iter2 = data_set.create_dict_iterator(num_epochs=1, output_numpy=True)
    assert (next(dataset_iter2)["array_a"] == data[0]["array_a"]).all()
    assert (next(dataset_iter2)["array_a"] == data[1]["array_a"]).all()

    dataset_iter3 = data_set.create_dict_iterator(num_epochs=1, output_numpy=True)
    assert (next(dataset_iter3)["array_a"] == data[0]["array_a"]).all()
    assert (next(dataset_iter3)["array_a"] == data[1]["array_a"]).all()
    assert (next(dataset_iter3)["array_a"] == data[2]["array_a"]).all()

    assert (next(dataset_iter2)["array_a"] == data[2]["array_a"]).all()
    assert (next(dataset_iter2)["array_a"] == data[3]["array_a"]).all()

    dataset_iter4 = data_set.create_dict_iterator(num_epochs=1, output_numpy=True)
    assert (next(dataset_iter4)["array_a"] == data[0]["array_a"]).all()
    assert (next(dataset_iter4)["array_a"] == data[1]["array_a"]).all()
    assert (next(dataset_iter4)["array_a"] == data[2]["array_a"]).all()

    assert (next(dataset_iter3)["array_a"] == data[3]["array_a"]).all()
    assert (next(dataset_iter3)["array_a"] == data[4]["array_a"]).all()
    assert (next(dataset_iter3)["array_a"] == data[5]["array_a"]).all()


def test_minddataset_multi_files_with_empty_one(add_and_remove_three_nlp_file_with_sample_10_0_5):
    """
    Feature: MindDataset
    Description: Test for with multi mindrecord files but there is one empty
    Expectation: Output is equal to the expected output
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    paths = [file_name + "_with_sample_10",
             file_name + "_with_sample_0",
             file_name + "_with_sample_5"]
    epoch_size = 3

    # with distribute
    def read_multi_mindrecord_files_with_distribute(file_names):
        data_set = ds.MindDataset(dataset_files=file_names,
                                  num_shards=4,
                                  shard_id=1)
        assert data_set.get_dataset_size() == 4
        dataset_iter = data_set.create_dict_iterator(num_epochs=epoch_size, output_numpy=True)
        for epoch in range(epoch_size):  # 3 epoch
            num_iter = 0
            for item in dataset_iter:
                num_iter += 1
            assert num_iter == 4

    read_multi_mindrecord_files_with_distribute([paths[0], paths[1], paths[2]])
    read_multi_mindrecord_files_with_distribute([paths[1], paths[0], paths[2]])
    read_multi_mindrecord_files_with_distribute([paths[0], paths[2], paths[1]])

    # with non-distribute
    def read_multi_mindrecord_files(file_names):
        data_set = ds.MindDataset(dataset_files=file_names)
        assert data_set.get_dataset_size() == 15
        dataset_iter = data_set.create_dict_iterator(num_epochs=epoch_size, output_numpy=True)
        for epoch in range(epoch_size):  # 3 epoch
            num_iter = 0
            for item in dataset_iter:
                num_iter += 1
            assert num_iter == 15

    read_multi_mindrecord_files([paths[0], paths[1], paths[2]])
    read_multi_mindrecord_files([paths[1], paths[0], paths[2]])
    read_multi_mindrecord_files([paths[0], paths[2], paths[1]])

if __name__ == '__main__':
    test_nlp_compress_data(add_and_remove_nlp_compress_file)
    test_nlp_compress_data_old_version(add_and_remove_nlp_compress_file)
    test_cv_minddataset_writer_tutorial()
    test_cv_minddataset_partition_tutorial(add_and_remove_cv_file)
    test_cv_minddataset_partition_num_samples_0(add_and_remove_cv_file)
    test_cv_minddataset_partition_num_samples_1(add_and_remove_cv_file)
    test_cv_minddataset_partition_num_samples_2(add_and_remove_cv_file)
    test_cv_minddataset_partition_tutorial_check_shuffle_result(add_and_remove_cv_file)
    test_cv_minddataset_partition_tutorial_check_whole_reshuffle_result_per_epoch(add_and_remove_cv_file)
    test_cv_minddataset_check_shuffle_result(add_and_remove_cv_file)
    test_cv_minddataset_dataset_size(add_and_remove_cv_file)
    test_cv_minddataset_repeat_reshuffle(add_and_remove_cv_file)
    test_cv_minddataset_batch_size_larger_than_records(add_and_remove_cv_file)
    test_cv_minddataset_issue_888(add_and_remove_cv_file)
    test_cv_minddataset_blockreader_tutorial(add_and_remove_cv_file)
    test_cv_minddataset_blockreader_some_field_not_in_index_tutorial(add_and_remove_cv_file)
    test_cv_minddataset_reader_file_list(add_and_remove_cv_file)
    test_cv_minddataset_reader_one_partition(add_and_remove_cv_file)
    test_cv_minddataset_reader_two_dataset(add_and_remove_cv_file)
    test_cv_minddataset_reader_two_dataset_partition(add_and_remove_cv_file)
    test_cv_minddataset_reader_basic_tutorial(add_and_remove_cv_file)
    test_nlp_minddataset_reader_basic_tutorial(add_and_remove_cv_file)
    test_cv_minddataset_reader_basic_tutorial_5_epoch(add_and_remove_cv_file)
    test_cv_minddataset_reader_basic_tutorial_5_epoch_with_batch(add_and_remove_cv_file)
    test_cv_minddataset_reader_no_columns(add_and_remove_cv_file)
    test_cv_minddataset_reader_repeat_tutorial(add_and_remove_cv_file)
    test_write_with_multi_bytes_and_array_and_read_by_MindDataset()
    test_write_with_multi_bytes_and_MindDataset()
    test_write_with_multi_array_and_MindDataset()
    test_numpy_generic()
    test_write_with_float32_float64_float32_array_float64_array_and_MindDataset()
    test_shuffle_with_global_infile_files(create_multi_mindrecord_files)
    test_distributed_shuffle_with_global_infile_files(create_multi_mindrecord_files)
    test_distributed_shuffle_with_multi_epochs(create_multi_mindrecord_files)
    test_field_is_null_numpy()
    test_for_loop_dataset_iterator(add_and_remove_nlp_compress_file)
