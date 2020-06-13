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
This is the test module for mindrecord
"""
import collections
import json
import numpy as np
import os
import pytest
import re
import string

import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.c_transforms as vision
from mindspore import log as logger
from mindspore.dataset.transforms.vision import Inter
from mindspore.mindrecord import FileWriter

FILES_NUM = 4
CV_FILE_NAME = "../data/mindrecord/imagenet.mindrecord"
CV1_FILE_NAME = "../data/mindrecord/imagenet1.mindrecord"
CV2_FILE_NAME = "../data/mindrecord/imagenet2.mindrecord"
CV_DIR_NAME = "../data/mindrecord/testImageNetData"
NLP_FILE_NAME = "../data/mindrecord/aclImdb.mindrecord"
NLP_FILE_POS = "../data/mindrecord/testAclImdbData/pos"
NLP_FILE_VOCAB = "../data/mindrecord/testAclImdbData/vocab.txt"


@pytest.fixture
def add_and_remove_cv_file():
    """add/remove cv file"""
    paths = ["{}{}".format(CV_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    for x in paths:
        os.remove("{}".format(x)) if os.path.exists("{}".format(x)) else None
        os.remove("{}.db".format(x)) if os.path.exists(
            "{}.db".format(x)) else None
    writer = FileWriter(CV_FILE_NAME, FILES_NUM)
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
    for x in paths:
        os.remove("{}".format(x))
        os.remove("{}.db".format(x))


@pytest.fixture
def add_and_remove_nlp_file():
    """add/remove nlp file"""
    paths = ["{}{}".format(NLP_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    for x in paths:
        if os.path.exists("{}".format(x)):
            os.remove("{}".format(x))
        if os.path.exists("{}.db".format(x)):
            os.remove("{}.db".format(x))
    writer = FileWriter(NLP_FILE_NAME, FILES_NUM)
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
    for x in paths:
        os.remove("{}".format(x))
        os.remove("{}.db".format(x))

def test_cv_minddataset_reader_basic_padded_samples(add_and_remove_cv_file):
    """tutorial for cv minderdataset."""
    columns_list = ["label", "file_name", "data"]

    data = get_data(CV_DIR_NAME)
    padded_sample = data[0]
    padded_sample['label'] = -1
    padded_sample['file_name'] = 'dummy.jpg'
    num_readers = 4
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers, padded_sample=padded_sample, num_padded=5)
    assert data_set.get_dataset_size() == 15
    num_iter = 0
    num_padded_iter = 0
    for item in data_set.create_dict_iterator():
        logger.info("-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info("-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info("-------------- item[label]: {} ----------------------------".format(item["label"]))
        if item['label'] == -1:
            num_padded_iter += 1
            assert item['file_name'] == bytes(padded_sample['file_name'],
                    encoding='utf8')
            assert item['label'] == padded_sample['label']
            assert (item['data'] == np.array(list(padded_sample['data']))).all()
        num_iter += 1 
    assert num_padded_iter == 5
    assert num_iter == 15


def test_cv_minddataset_partition_padded_samples(add_and_remove_cv_file):
    """tutorial for cv minddataset."""
    columns_list = ["data", "file_name", "label"]

    data = get_data(CV_DIR_NAME)
    padded_sample = data[0]
    padded_sample['label'] = -2
    padded_sample['file_name'] = 'dummy.jpg'
    num_readers = 4

    def partitions(num_shards, num_padded, dataset_size):
        num_padded_iter = 0
        num_iter = 0
        for partition_id in range(num_shards):
            data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
                                      num_shards=num_shards,
                                      shard_id=partition_id,
                                      padded_sample=padded_sample,
                                      num_padded=num_padded)
            assert data_set.get_dataset_size() == dataset_size
            for item in data_set.create_dict_iterator():
                logger.info("-------------- partition : {} ------------------------".format(partition_id))
                logger.info("-------------- len(item[data]): {} ------------------------".format(len(item["data"])))
                logger.info("-------------- item[data]: {} -----------------------------".format(item["data"]))
                logger.info("-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
                logger.info("-------------- item[label]: {} -----------------------".format(item["label"]))
                if item['label'] == -2:
                    num_padded_iter += 1
                    assert item['file_name'] == bytes(padded_sample['file_name'], encoding='utf8')
                    assert item['label'] == padded_sample['label']
                    assert (item['data'] == np.array(list(padded_sample['data']))).all()
                num_iter += 1
        assert num_padded_iter == num_padded
        return num_iter == dataset_size * num_shards

    partitions(4, 2, 3)
    partitions(5, 5, 3)
    partitions(9, 8, 2)

def test_cv_minddataset_partition_padded_samples_multi_epoch(add_and_remove_cv_file):
    """tutorial for cv minddataset."""
    columns_list = ["data", "file_name", "label"]

    data = get_data(CV_DIR_NAME)
    padded_sample = data[0]
    padded_sample['label'] = -2
    padded_sample['file_name'] = 'dummy.jpg'
    num_readers = 4

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
            data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
                                      num_shards=num_shards,
                                      shard_id=partition_id,
                                      padded_sample=padded_sample,
                                      num_padded=num_padded)
            assert data_set.get_dataset_size() == dataset_size
            data_set = data_set.repeat(repeat_size)
            local_index = 0
            for item in data_set.create_dict_iterator():
                logger.info("-------------- partition : {} ------------------------".format(partition_id))
                logger.info("-------------- len(item[data]): {} ------------------------".format(len(item["data"])))
                logger.info("-------------- item[data]: {} -----------------------------".format(item["data"]))
                logger.info("-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
                logger.info("-------------- item[label]: {} -----------------------".format(item["label"]))
                if item['label'] == -2:
                    num_padded_iter += 1
                    assert item['file_name'] == bytes(padded_sample['file_name'], encoding='utf8')
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

def test_cv_minddataset_partition_padded_samples_no_dividsible(add_and_remove_cv_file):
    """tutorial for cv minddataset."""
    columns_list = ["data", "file_name", "label"]

    data = get_data(CV_DIR_NAME)
    padded_sample = data[0]
    padded_sample['label'] = -2
    padded_sample['file_name'] = 'dummy.jpg'
    num_readers = 4

    def partitions(num_shards, num_padded):
        for partition_id in range(num_shards):
            data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
                                      num_shards=num_shards,
                                      shard_id=partition_id,
                                      padded_sample=padded_sample,
                                      num_padded=num_padded)
            num_iter = 0
            for item in data_set.create_dict_iterator():
                num_iter += 1
        return num_iter

    with pytest.raises(RuntimeError):
        partitions(4, 1)

def test_cv_minddataset_partition_padded_samples_dataset_size_no_divisible(add_and_remove_cv_file):
    columns_list = ["data", "file_name", "label"]

    data = get_data(CV_DIR_NAME)
    padded_sample = data[0]
    padded_sample['label'] = -2
    padded_sample['file_name'] = 'dummy.jpg'
    num_readers = 4

    def partitions(num_shards, num_padded):
        for partition_id in range(num_shards):
            data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
                                      num_shards=num_shards,
                                      shard_id=partition_id,
                                      padded_sample=padded_sample,
                                      num_padded=num_padded)
            with pytest.raises(RuntimeError):
                data_set.get_dataset_size() == 3
    partitions(4, 1)

def test_cv_minddataset_partition_padded_samples_no_equal_column_list(add_and_remove_cv_file):
    columns_list = ["data", "file_name", "label"]

    data = get_data(CV_DIR_NAME)
    padded_sample = data[0]
    padded_sample.pop('label', None)
    padded_sample['file_name'] = 'dummy.jpg'
    num_readers = 4

    def partitions(num_shards, num_padded):
        for partition_id in range(num_shards):
            data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
                                      num_shards=num_shards,
                                      shard_id=partition_id,
                                      padded_sample=padded_sample,
                                      num_padded=num_padded)
        for item in data_set.create_dict_iterator():
            logger.info("-------------- partition : {} ------------------------".format(partition_id))
            logger.info("-------------- len(item[data]): {} ------------------------".format(len(item["data"])))
            logger.info("-------------- item[data]: {} -----------------------------".format(item["data"]))
            logger.info("-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
    with pytest.raises(Exception, match="padded_sample cannot match columns_list."):
        partitions(4, 2)

def test_cv_minddataset_partition_padded_samples_no_column_list(add_and_remove_cv_file):
    data = get_data(CV_DIR_NAME)
    padded_sample = data[0]
    padded_sample['label'] = -2
    padded_sample['file_name'] = 'dummy.jpg'
    num_readers = 4

    def partitions(num_shards, num_padded):
        for partition_id in range(num_shards):
            data_set = ds.MindDataset(CV_FILE_NAME + "0", None, num_readers,
                                      num_shards=num_shards,
                                      shard_id=partition_id,
                                      padded_sample=padded_sample,
                                      num_padded=num_padded)
        for item in data_set.create_dict_iterator():
            logger.info("-------------- partition : {} ------------------------".format(partition_id))
            logger.info("-------------- len(item[data]): {} ------------------------".format(len(item["data"])))
            logger.info("-------------- item[data]: {} -----------------------------".format(item["data"]))
            logger.info("-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
    with pytest.raises(Exception, match="padded_sample is specified and requires columns_list as well."):
        partitions(4, 2)

def test_cv_minddataset_partition_padded_samples_no_num_padded(add_and_remove_cv_file):
    columns_list = ["data", "file_name", "label"]
    data = get_data(CV_DIR_NAME)
    padded_sample = data[0]
    padded_sample['file_name'] = 'dummy.jpg'
    num_readers = 4

    def partitions(num_shards, num_padded):
        for partition_id in range(num_shards):
            data_set = ds.MindDataset(CV_FILE_NAME + "0", None, num_readers,
                                      num_shards=num_shards,
                                      shard_id=partition_id,
                                      padded_sample=padded_sample)
        for item in data_set.create_dict_iterator():
            logger.info("-------------- partition : {} ------------------------".format(partition_id))
            logger.info("-------------- len(item[data]): {} ------------------------".format(len(item["data"])))
            logger.info("-------------- item[data]: {} -----------------------------".format(item["data"]))
            logger.info("-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
    with pytest.raises(Exception, match="padded_sample is specified and requires num_padded as well."):
        partitions(4, 2)

def test_cv_minddataset_partition_padded_samples_no_padded_samples(add_and_remove_cv_file):
    columns_list = ["data", "file_name", "label"]
    data = get_data(CV_DIR_NAME)
    padded_sample = data[0]
    padded_sample['file_name'] = 'dummy.jpg'
    num_readers = 4

    def partitions(num_shards, num_padded):
        for partition_id in range(num_shards):
            data_set = ds.MindDataset(CV_FILE_NAME + "0", None, num_readers,
                                      num_shards=num_shards,
                                      shard_id=partition_id,
                                      num_padded=num_padded)
        for item in data_set.create_dict_iterator():
            logger.info("-------------- partition : {} ------------------------".format(partition_id))
            logger.info("-------------- len(item[data]): {} ------------------------".format(len(item["data"])))
            logger.info("-------------- item[data]: {} -----------------------------".format(item["data"]))
            logger.info("-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
    with pytest.raises(Exception, match="num_padded is specified but padded_sample is not."):
        partitions(4, 2)



def test_nlp_minddataset_reader_basic_padded_samples(add_and_remove_nlp_file):
    columns_list = ["input_ids", "id", "rating"]

    data = [x for x in get_nlp_data(NLP_FILE_POS, NLP_FILE_VOCAB, 10)]
    padded_sample = data[0]
    padded_sample['id'] = "-1"
    padded_sample['input_ids'] = np.array([-1,-1,-1,-1], dtype=np.int64)
    padded_sample['rating'] = 1.0
    num_readers = 4

    def partitions(num_shards, num_padded, dataset_size):
        num_padded_iter = 0
        num_iter = 0
        for partition_id in range(num_shards):
            data_set = ds.MindDataset(NLP_FILE_NAME + "0", columns_list, num_readers,
                                      num_shards=num_shards,
                                      shard_id=partition_id,
                                      padded_sample=padded_sample,
                                      num_padded=num_padded)
            assert data_set.get_dataset_size() == dataset_size
            for item in data_set.create_dict_iterator():
                logger.info("-------------- item[id]: {} ------------------------".format(item["id"]))
                logger.info("-------------- item[rating]: {} --------------------".format(item["rating"]))
                logger.info("-------------- item[input_ids]: {}, shape: {} -----------------".format(item["input_ids"], item["input_ids"].shape))
                if item['id'] == bytes('-1', encoding='utf-8'):
                    num_padded_iter += 1
                    assert item['id'] == bytes(padded_sample['id'], encoding='utf-8')
                    assert (item['input_ids'] == padded_sample['input_ids']).all()
                    assert (item['rating'] == padded_sample['rating']).all()
                num_iter += 1
        assert num_padded_iter == num_padded
        assert num_iter == dataset_size * num_shards

    partitions(4, 6, 4)
    partitions(5, 5, 3)
    partitions(9, 8, 2)

def test_nlp_minddataset_reader_basic_padded_samples_multi_epoch(add_and_remove_nlp_file):
    columns_list = ["input_ids", "id", "rating"]

    data = [x for x in get_nlp_data(NLP_FILE_POS, NLP_FILE_VOCAB, 10)]
    padded_sample = data[0]
    padded_sample['id'] = "-1"
    padded_sample['input_ids'] = np.array([-1,-1,-1,-1], dtype=np.int64)
    padded_sample['rating'] = 1.0
    num_readers = 4
    repeat_size = 3

    def partitions(num_shards, num_padded, dataset_size):
        num_padded_iter = 0
        num_iter = 0

        for partition_id in range(num_shards):
            epoch1_shuffle_result = []
            epoch2_shuffle_result = []
            epoch3_shuffle_result = []
            data_set = ds.MindDataset(NLP_FILE_NAME + "0", columns_list, num_readers,
                                      num_shards=num_shards,
                                      shard_id=partition_id,
                                      padded_sample=padded_sample,
                                      num_padded=num_padded)
            assert data_set.get_dataset_size() == dataset_size
            data_set = data_set.repeat(repeat_size)

            local_index = 0
            for item in data_set.create_dict_iterator():
                logger.info("-------------- item[id]: {} ------------------------".format(item["id"]))
                logger.info("-------------- item[rating]: {} --------------------".format(item["rating"]))
                logger.info("-------------- item[input_ids]: {}, shape: {} -----------------".format(item["input_ids"], item["input_ids"].shape))
                if item['id'] == bytes('-1', encoding='utf-8'):
                    num_padded_iter += 1
                    assert item['id'] == bytes(padded_sample['id'], encoding='utf-8')
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
