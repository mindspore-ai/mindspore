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
This is the test module for mindrecord
"""
import collections
import os
import re
import string

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore import log as logger
from mindspore.mindrecord import FileWriter

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
            os.remove("{}".format(x)) if os.path.exists("{}".format(x)) else None
            os.remove("{}.db".format(x)) if os.path.exists(
                "{}.db".format(x)) else None
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


@pytest.mark.usefixtures("add_and_remove_cv_file")
def test_cv_minddataset_reader_basic_padded_samples():
    """
    Feature: MindDataset
    Description: Test basic read on MindDataset with padded_sample
    Expectation: Output is equal to the expected output
    """
    columns_list = ["label", "file_name", "data"]

    data = get_data(CV_DIR_NAME)
    padded_sample = data[0]
    padded_sample['label'] = -1
    padded_sample['file_name'] = 'dummy.jpg'
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers, padded_sample=padded_sample, num_padded=5)
    assert data_set.get_dataset_size() == 15
    num_iter = 0
    num_padded_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info("-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info("-------------- item[label]: {} ----------------------------".format(item["label"]))
        if item['label'] == -1:
            num_padded_iter += 1
            assert item['file_name'] == padded_sample['file_name']
            assert item['label'] == padded_sample['label']
            assert (item['data'] == np.array(list(padded_sample['data']))).all()
        num_iter += 1
    assert num_padded_iter == 5
    assert num_iter == 15


@pytest.mark.usefixtures("add_and_remove_cv_file")
def test_cv_minddataset_reader_basic_padded_samples_type_cast():
    """
    Feature: MindDataset
    Description: Test basic read on MindDataset with padded_sample which file_name requires type cast
    Expectation: Output is equal to the expected output
    """
    columns_list = ["label", "file_name", "data"]

    data = get_data(CV_DIR_NAME)
    padded_sample = data[0]
    padded_sample['label'] = -1
    padded_sample['file_name'] = "99999"
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data_set = ds.MindDataset(file_name + "0", columns_list, num_readers, padded_sample=padded_sample, num_padded=5)
    assert data_set.get_dataset_size() == 15
    num_iter = 0
    num_padded_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info("-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info("-------------- item[label]: {} ----------------------------".format(item["label"]))
        if item['label'] == -1:
            num_padded_iter += 1
            assert item['file_name'] == padded_sample['file_name']
            assert item['label'] == padded_sample['label']
            assert (item['data'] == np.array(list(padded_sample['data']))).all()
        num_iter += 1
    assert num_padded_iter == 5
    assert num_iter == 15


@pytest.mark.usefixtures("add_and_remove_cv_file")
def test_cv_minddataset_partition_padded_samples():
    """
    Feature: MindDataset
    Description: Test read on MindDataset with padded_sample and partition (num_shards and shard_id)
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]

    data = get_data(CV_DIR_NAME)
    padded_sample = data[0]
    padded_sample['label'] = -2
    padded_sample['file_name'] = 'dummy.jpg'
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    def partitions(num_shards, num_padded, dataset_size):
        num_padded_iter = 0
        num_iter = 0
        for partition_id in range(num_shards):
            data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                                      num_shards=num_shards,
                                      shard_id=partition_id,
                                      padded_sample=padded_sample,
                                      num_padded=num_padded)
            assert data_set.get_dataset_size() == dataset_size
            for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
                logger.info("-------------- partition : {} ------------------------".format(partition_id))
                logger.info("-------------- len(item[data]): {} ------------------------".format(len(item["data"])))
                logger.info("-------------- item[data]: {} -----------------------------".format(item["data"]))
                logger.info("-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
                logger.info("-------------- item[label]: {} -----------------------".format(item["label"]))
                if item['label'] == -2:
                    num_padded_iter += 1
                    assert item['file_name'] == padded_sample['file_name']
                    assert item['label'] == padded_sample['label']
                    assert (item['data'] == np.array(list(padded_sample['data']))).all()
                num_iter += 1
        assert num_padded_iter == num_padded
        return num_iter == dataset_size * num_shards

    partitions(4, 2, 3)
    partitions(5, 5, 3)
    partitions(9, 8, 2)


@pytest.mark.usefixtures("add_and_remove_cv_file")
def test_cv_minddataset_partition_padded_samples_multi_epoch():
    """
    Feature: MindDataset
    Description: Test read on MindDataset with padded_sample and partition (num_shards and shard_id),
        performed under multiple epochs
    Expectation: Output is equal to the expected output
    """
    columns_list = ["data", "file_name", "label"]

    data = get_data(CV_DIR_NAME)
    padded_sample = data[0]
    padded_sample['label'] = -2
    padded_sample['file_name'] = 'dummy.jpg'
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    def partitions(num_shards, num_padded, dataset_size):
        repeat_size = 5
        num_padded_iter = 0
        num_iter = 0
        for partition_id in range(num_shards):
            epoch1_shuffle_result = []
            epoch2_shuffle_result = []
            epoch3_shuffle_result = []
            epoch4_shuffle_result = []
            epoch5_shuffle_result = []
            data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                                      num_shards=num_shards,
                                      shard_id=partition_id,
                                      padded_sample=padded_sample,
                                      num_padded=num_padded)
            assert data_set.get_dataset_size() == dataset_size
            data_set = data_set.repeat(repeat_size)
            local_index = 0
            for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
                logger.info("-------------- partition : {} ------------------------".format(partition_id))
                logger.info("-------------- len(item[data]): {} ------------------------".format(len(item["data"])))
                logger.info("-------------- item[data]: {} -----------------------------".format(item["data"]))
                logger.info("-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
                logger.info("-------------- item[label]: {} -----------------------".format(item["label"]))
                if item['label'] == -2:
                    num_padded_iter += 1
                    assert item['file_name'] == padded_sample['file_name']
                    assert item['label'] == padded_sample['label']
                    assert (item['data'] == np.array(list(padded_sample['data']))).all()
                if local_index < dataset_size:
                    epoch1_shuffle_result.append(item["file_name"])
                elif local_index < dataset_size * 2:
                    epoch2_shuffle_result.append(item["file_name"])
                elif local_index < dataset_size * 3:
                    epoch3_shuffle_result.append(item["file_name"])
                elif local_index < dataset_size * 4:
                    epoch4_shuffle_result.append(item["file_name"])
                elif local_index < dataset_size * 5:
                    epoch5_shuffle_result.append(item["file_name"])
                local_index += 1
                num_iter += 1
            assert len(epoch1_shuffle_result) == dataset_size
            assert len(epoch2_shuffle_result) == dataset_size
            assert len(epoch3_shuffle_result) == dataset_size
            assert len(epoch4_shuffle_result) == dataset_size
            assert len(epoch5_shuffle_result) == dataset_size
            assert local_index == dataset_size * repeat_size

            # When dataset_size is equal to 2, too high probability is the same result after shuffle operation
            if dataset_size > 2:
                assert epoch1_shuffle_result != epoch2_shuffle_result
                assert epoch2_shuffle_result != epoch3_shuffle_result
                assert epoch3_shuffle_result != epoch4_shuffle_result
                assert epoch4_shuffle_result != epoch5_shuffle_result
        assert num_padded_iter == num_padded * repeat_size
        assert num_iter == dataset_size * num_shards * repeat_size

    partitions(4, 2, 3)
    partitions(5, 5, 3)
    partitions(9, 8, 2)


@pytest.mark.usefixtures("add_and_remove_cv_file")
def test_cv_minddataset_partition_padded_samples_no_dividsible():
    """
    Feature: MindDataset
    Description: Test read on MindDataset with padded_sample and partition (num_shards and shard_id),
        where num_padded is not divisible
    Expectation: Error is raised as expected
    """
    columns_list = ["data", "file_name", "label"]

    data = get_data(CV_DIR_NAME)
    padded_sample = data[0]
    padded_sample['label'] = -2
    padded_sample['file_name'] = 'dummy.jpg'
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    def partitions(num_shards, num_padded):
        for partition_id in range(num_shards):
            data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                                      num_shards=num_shards,
                                      shard_id=partition_id,
                                      padded_sample=padded_sample,
                                      num_padded=num_padded)
            num_iter = 0
            for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
                num_iter += 1
        return num_iter

    with pytest.raises(RuntimeError):
        partitions(4, 1)


@pytest.mark.usefixtures("add_and_remove_cv_file")
def test_cv_minddataset_partition_padded_samples_dataset_size_no_divisible():
    """
    Feature: MindDataset
    Description: Test get_dataset_size during MindDataset read with padded_sample and partition
        (num_shards and shard_id), where num_padded is not divisible
    Expectation: Error is raised as expected
    """
    columns_list = ["data", "file_name", "label"]

    data = get_data(CV_DIR_NAME)
    padded_sample = data[0]
    padded_sample['label'] = -2
    padded_sample['file_name'] = 'dummy.jpg'
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    def partitions(num_shards, num_padded):
        for partition_id in range(num_shards):
            data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                                      num_shards=num_shards,
                                      shard_id=partition_id,
                                      padded_sample=padded_sample,
                                      num_padded=num_padded)
            with pytest.raises(RuntimeError):
                data_set.get_dataset_size() == 3

    partitions(4, 1)


@pytest.mark.usefixtures("add_and_remove_cv_file")
def test_cv_minddataset_partition_padded_samples_no_equal_column_list():
    """
    Feature: MindDataset
    Description: Test read MindDataset with padded_sample and partition
        (num_shards and shard_id), where padded_sample does not match columns_list
    Expectation: Error is raised as expected
    """
    columns_list = ["data", "file_name", "label"]

    data = get_data(CV_DIR_NAME)
    padded_sample = data[0]
    padded_sample.pop('label', None)
    padded_sample['file_name'] = 'dummy.jpg'
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    def partitions(num_shards, num_padded):
        for partition_id in range(num_shards):
            data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                                      num_shards=num_shards,
                                      shard_id=partition_id,
                                      padded_sample=padded_sample,
                                      num_padded=num_padded)
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            logger.info("-------------- partition : {} ------------------------".format(partition_id))
            logger.info("-------------- len(item[data]): {} ------------------------".format(len(item["data"])))
            logger.info("-------------- item[data]: {} -----------------------------".format(item["data"]))
            logger.info("-------------- item[file_name]: {} ------------------------".format(item["file_name"]))

    with pytest.raises(Exception, match="padded_sample cannot match columns_list."):
        partitions(4, 2)


@pytest.mark.usefixtures("add_and_remove_cv_file")
def test_cv_minddataset_partition_padded_samples_no_column_list():
    """
    Feature: MindDataset
    Description: Test read MindDataset with padded_sample and partition
        (num_shards and shard_id), where there is no columns_list
    Expectation: Error is raised as expected
    """
    data = get_data(CV_DIR_NAME)
    padded_sample = data[0]
    padded_sample['label'] = -2
    padded_sample['file_name'] = 'dummy.jpg'
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    def partitions(num_shards, num_padded):
        for partition_id in range(num_shards):
            data_set = ds.MindDataset(file_name + "0", None, num_readers,
                                      num_shards=num_shards,
                                      shard_id=partition_id,
                                      padded_sample=padded_sample,
                                      num_padded=num_padded)
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            logger.info("-------------- partition : {} ------------------------".format(partition_id))
            logger.info("-------------- len(item[data]): {} ------------------------".format(len(item["data"])))
            logger.info("-------------- item[data]: {} -----------------------------".format(item["data"]))
            logger.info("-------------- item[file_name]: {} ------------------------".format(item["file_name"]))

    with pytest.raises(Exception, match="padded_sample is specified and requires columns_list as well."):
        partitions(4, 2)


@pytest.mark.usefixtures("add_and_remove_cv_file")
def test_cv_minddataset_partition_padded_samples_no_num_padded():
    """
    Feature: MindDataset
    Description: Test read MindDataset with padded_sample and partition
        (num_shards and shard_id), where there is no num_padded
    Expectation: Error is raised as expected
    """
    columns_list = ["data", "file_name", "label"]
    data = get_data(CV_DIR_NAME)
    padded_sample = data[0]
    padded_sample['file_name'] = 'dummy.jpg'
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    def partitions(num_shards, num_padded):
        for partition_id in range(num_shards):
            data_set = ds.MindDataset(file_name + "0", None, num_readers,
                                      num_shards=num_shards,
                                      shard_id=partition_id,
                                      padded_sample=padded_sample)
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            logger.info("-------------- partition : {} ------------------------".format(partition_id))
            logger.info("-------------- len(item[data]): {} ------------------------".format(len(item["data"])))
            logger.info("-------------- item[data]: {} -----------------------------".format(item["data"]))
            logger.info("-------------- item[file_name]: {} ------------------------".format(item["file_name"]))

    with pytest.raises(Exception, match="padded_sample is specified and requires num_padded as well."):
        partitions(4, 2)


@pytest.mark.usefixtures("add_and_remove_cv_file")
def test_cv_minddataset_partition_padded_samples_no_padded_samples():
    """
    Feature: MindDataset
    Description: Test read MindDataset with padded_sample and partition
        (num_shards and shard_id), where there is no padded_sample
    Expectation: Error is raised as expected
    """
    columns_list = ["data", "file_name", "label"]
    data = get_data(CV_DIR_NAME)
    padded_sample = data[0]
    padded_sample['file_name'] = 'dummy.jpg'
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    def partitions(num_shards, num_padded):
        for partition_id in range(num_shards):
            data_set = ds.MindDataset(file_name + "0", None, num_readers,
                                      num_shards=num_shards,
                                      shard_id=partition_id,
                                      num_padded=num_padded)
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            logger.info("-------------- partition : {} ------------------------".format(partition_id))
            logger.info("-------------- len(item[data]): {} ------------------------".format(len(item["data"])))
            logger.info("-------------- item[data]: {} -----------------------------".format(item["data"]))
            logger.info("-------------- item[file_name]: {} ------------------------".format(item["file_name"]))

    with pytest.raises(Exception, match="num_padded is specified but padded_sample is not."):
        partitions(4, 2)


@pytest.mark.usefixtures("add_and_remove_nlp_file")
def test_nlp_minddataset_reader_basic_padded_samples():
    """
    Feature: MindDataset
    Description: Test basic read MindDataset with padded_sample from raw data of aclImdb dataset
    Expectation: Output is equal to the expected output
    """
    columns_list = ["input_ids", "id", "rating"]

    data = [x for x in get_nlp_data(NLP_FILE_POS, NLP_FILE_VOCAB, 10)]
    padded_sample = data[0]
    padded_sample['id'] = "-1"
    padded_sample['input_ids'] = np.array([-1, -1, -1, -1], dtype=np.int64)
    padded_sample['rating'] = 1.0
    num_readers = 4
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    def partitions(num_shards, num_padded, dataset_size):
        num_padded_iter = 0
        num_iter = 0
        for partition_id in range(num_shards):
            data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                                      num_shards=num_shards,
                                      shard_id=partition_id,
                                      padded_sample=padded_sample,
                                      num_padded=num_padded)
            assert data_set.get_dataset_size() == dataset_size
            for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
                logger.info("-------------- item[id]: {} ------------------------".format(item["id"]))
                logger.info("-------------- item[rating]: {} --------------------".format(item["rating"]))
                logger.info("-------------- item[input_ids]: {}, shape: {} -----------------".format(
                    item["input_ids"],
                    item["input_ids"].shape))
                if item['id'] == '-1':
                    num_padded_iter += 1
                    assert item['id'] == padded_sample['id']
                    assert (item['input_ids'] == padded_sample['input_ids']).all()
                    assert (item['rating'] == padded_sample['rating']).all()
                num_iter += 1
        assert num_padded_iter == num_padded
        assert num_iter == dataset_size * num_shards

    partitions(4, 6, 4)
    partitions(5, 5, 3)
    partitions(9, 8, 2)


@pytest.mark.usefixtures("add_and_remove_nlp_file")
def test_nlp_minddataset_reader_basic_padded_samples_multi_epoch():
    """
    Feature: MindDataset
    Description: Test basic read MindDataset with padded_sample from raw data of aclImdb dataset under multiple epochs
    Expectation: Output is equal to the expected output
    """
    columns_list = ["input_ids", "id", "rating"]

    data = [x for x in get_nlp_data(NLP_FILE_POS, NLP_FILE_VOCAB, 10)]
    padded_sample = data[0]
    padded_sample['id'] = "-1"
    padded_sample['input_ids'] = np.array([-1, -1, -1, -1], dtype=np.int64)
    padded_sample['rating'] = 1.0
    num_readers = 4
    repeat_size = 3
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    def partitions(num_shards, num_padded, dataset_size):
        num_padded_iter = 0
        num_iter = 0

        for partition_id in range(num_shards):
            epoch1_shuffle_result = []
            epoch2_shuffle_result = []
            epoch3_shuffle_result = []
            data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                                      num_shards=num_shards,
                                      shard_id=partition_id,
                                      padded_sample=padded_sample,
                                      num_padded=num_padded)
            assert data_set.get_dataset_size() == dataset_size
            data_set = data_set.repeat(repeat_size)

            local_index = 0
            for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
                logger.info("-------------- item[id]: {} ------------------------".format(item["id"]))
                logger.info("-------------- item[rating]: {} --------------------".format(item["rating"]))
                logger.info("-------------- item[input_ids]: {}, shape: {} -----------------".format(
                    item["input_ids"],
                    item["input_ids"].shape))
                if item['id'] == '-1':
                    num_padded_iter += 1
                    assert item['id'] == padded_sample['id']
                    assert (item['input_ids'] == padded_sample['input_ids']).all()
                    assert (item['rating'] == padded_sample['rating']).all()

                if local_index < dataset_size:
                    epoch1_shuffle_result.append(item['id'])
                elif local_index < dataset_size * 2:
                    epoch2_shuffle_result.append(item['id'])
                elif local_index < dataset_size * 3:
                    epoch3_shuffle_result.append(item['id'])
                local_index += 1
                num_iter += 1
            assert len(epoch1_shuffle_result) == dataset_size
            assert len(epoch2_shuffle_result) == dataset_size
            assert len(epoch3_shuffle_result) == dataset_size
            assert local_index == dataset_size * repeat_size

            # When dataset_size is equal to 2, too high probability is the same result after shuffle operation
            if dataset_size > 2:
                assert epoch1_shuffle_result != epoch2_shuffle_result
                assert epoch2_shuffle_result != epoch3_shuffle_result
        assert num_padded_iter == num_padded * repeat_size
        assert num_iter == dataset_size * num_shards * repeat_size

    partitions(4, 6, 4)
    partitions(5, 5, 3)
    partitions(9, 8, 2)


@pytest.mark.usefixtures("add_and_remove_nlp_file")
def test_nlp_minddataset_reader_basic_padded_samples_check_whole_reshuffle_result_per_epoch():
    """
    Feature: MindDataset
    Description: Test basic read MindDataset with padded_sample from raw data of aclImdb dataset
        by checking whole result_per_epoch to ensure there is no reshuffling
    Expectation: Output is equal to the expected output
    """
    original_seed = ds.config.get_seed()
    ds.config.set_seed(0)
    assert ds.config.get_seed() == 0

    columns_list = ["input_ids", "id", "rating"]

    padded_sample = {'id': "-1",
                     'input_ids': np.array([-1, -1, -1, -1], dtype=np.int64),
                     'rating': 1.0}
    num_readers = 4
    repeat_size = 3
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]

    def partitions(num_shards, num_padded, dataset_size):
        num_padded_iter = 0
        num_iter = 0

        epoch_result = [[["" for i in range(dataset_size)] for i in range(repeat_size)] for i in range(num_shards)]

        for partition_id in range(num_shards):
            data_set = ds.MindDataset(file_name + "0", columns_list, num_readers,
                                      num_shards=num_shards,
                                      shard_id=partition_id,
                                      padded_sample=padded_sample,
                                      num_padded=num_padded)
            assert data_set.get_dataset_size() == dataset_size
            data_set = data_set.repeat(repeat_size)
            inner_num_iter = 0
            for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
                logger.info("-------------- item[id]: {} ------------------------".format(item["id"]))
                logger.info("-------------- item[rating]: {} --------------------".format(item["rating"]))
                logger.info("-------------- item[input_ids]: {}, shape: {} -----------------"
                            .format(item["input_ids"], item["input_ids"].shape))
                if item['id'] == '-1':
                    num_padded_iter += 1
                    assert item['id'] == padded_sample.get('id')
                    assert (item['input_ids'] == padded_sample.get('input_ids')).all()
                    assert (item['rating'] == padded_sample.get('rating')).all()
                # save epoch result
                epoch_result[partition_id][int(inner_num_iter / dataset_size)][inner_num_iter % dataset_size] = item[
                    "id"]
                num_iter += 1
                inner_num_iter += 1
            assert epoch_result[partition_id][0] not in (epoch_result[partition_id][1], epoch_result[partition_id][2])
            assert epoch_result[partition_id][1] not in (epoch_result[partition_id][0], epoch_result[partition_id][2])
            assert epoch_result[partition_id][2] not in (epoch_result[partition_id][1], epoch_result[partition_id][0])
            if dataset_size > 2:
                epoch_result[partition_id][0].sort()
                epoch_result[partition_id][1].sort()
                epoch_result[partition_id][2].sort()
                assert epoch_result[partition_id][0] != epoch_result[partition_id][1]
                assert epoch_result[partition_id][1] != epoch_result[partition_id][2]
                assert epoch_result[partition_id][2] != epoch_result[partition_id][0]
        assert num_padded_iter == num_padded * repeat_size
        assert num_iter == dataset_size * num_shards * repeat_size

    partitions(4, 6, 4)
    partitions(5, 5, 3)
    partitions(9, 8, 2)

    # Restore config setting
    ds.config.set_seed(original_seed)


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
    for root, dirs, files in os.walk(dir_name):
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


if __name__ == '__main__':
    test_cv_minddataset_reader_basic_padded_samples()
    test_cv_minddataset_partition_padded_samples()
    test_cv_minddataset_partition_padded_samples_multi_epoch()
    test_cv_minddataset_partition_padded_samples_no_dividsible()
    test_cv_minddataset_partition_padded_samples_dataset_size_no_divisible()
    test_cv_minddataset_partition_padded_samples_no_equal_column_list()
    test_cv_minddataset_partition_padded_samples_no_column_list()
    test_cv_minddataset_partition_padded_samples_no_num_padded()
    test_cv_minddataset_partition_padded_samples_no_padded_samples()
    test_nlp_minddataset_reader_basic_padded_samples()
    test_nlp_minddataset_reader_basic_padded_samples_multi_epoch()
    test_nlp_minddataset_reader_basic_padded_samples_check_whole_reshuffle_result_per_epoch()
