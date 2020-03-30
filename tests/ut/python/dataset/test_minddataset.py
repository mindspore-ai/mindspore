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
This is the test module for mindrecord
"""
import collections
import json
import os
import re
import string

import mindspore.dataset.transforms.vision.c_transforms as vision
import numpy as np
import pytest
from mindspore._c_dataengine import InterpolationMode
from mindspore.dataset.transforms.vision import Inter
from mindspore import log as logger

import mindspore.dataset as ds
from mindspore.mindrecord import FileWriter

FILES_NUM = 4
CV_FILE_NAME = "../data/mindrecord/imagenet.mindrecord"
CV_DIR_NAME = "../data/mindrecord/testImageNetData"
NLP_FILE_NAME = "../data/mindrecord/aclImdb.mindrecord"
NLP_FILE_POS = "../data/mindrecord/testAclImdbData/pos"
NLP_FILE_VOCAB=  "../data/mindrecord/testAclImdbData/vocab.txt"

@pytest.fixture
def add_and_remove_cv_file():
    """add/remove cv file"""
    paths = ["{}{}".format(CV_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    for x in paths:
        os.remove("{}".format(x)) if os.path.exists("{}".format(x)) else None
        os.remove("{}.db".format(x)) if os.path.exists("{}.db".format(x)) else None
    writer = FileWriter(CV_FILE_NAME, FILES_NUM)
    data = get_data(CV_DIR_NAME)
    cv_schema_json = {"file_name": {"type": "string"}, "label": {"type": "int32"},
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
                                     "shape": [2,-1]}
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

def test_cv_minddataset_writer_tutorial():
    """tutorial for cv dataset writer."""
    paths = ["{}{}".format(CV_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    for x in paths:
        os.remove("{}".format(x)) if os.path.exists("{}".format(x)) else None
        os.remove("{}.db".format(x)) if os.path.exists("{}.db".format(x)) else None
    writer = FileWriter(CV_FILE_NAME, FILES_NUM)
    data = get_data(CV_DIR_NAME)
    cv_schema_json = {"file_name": {"type": "string"}, "label": {"type": "int32"},
                      "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    writer.write_raw_data(data)
    writer.commit()
    for x in paths:
        os.remove("{}".format(x))
        os.remove("{}.db".format(x))


def test_cv_minddataset_partition_tutorial(add_and_remove_cv_file):
    """tutorial for cv minddataset."""
    columns_list = ["data", "file_name", "label"]
    num_readers = 4

    def partitions(num_shards):
        for partition_id in range(num_shards):
            data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
                                      num_shards=num_shards, shard_id=partition_id)
            num_iter = 0
            for item in data_set.create_dict_iterator():
                logger.info("-------------- partition : {} ------------------------".format(partition_id))
                logger.info("-------------- item[label]: {} -----------------------".format(item["label"]))
                num_iter += 1
        return num_iter

    assert partitions(4) == 3
    assert partitions(5) == 2
    assert partitions(9) == 2


def test_cv_minddataset_dataset_size(add_and_remove_cv_file):
    """tutorial for cv minddataset."""
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers)
    assert data_set.get_dataset_size() == 10
    repeat_num = 2
    data_set = data_set.repeat(repeat_num)
    num_iter = 0
    for item in data_set.create_dict_iterator():
        logger.info("-------------- get dataset size {} -----------------".format(num_iter))
        logger.info("-------------- item[label]: {} ---------------------".format(item["label"]))
        logger.info("-------------- item[data]: {} ----------------------".format(item["data"]))
        num_iter += 1
    assert num_iter == 20
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
                              num_shards=4, shard_id=3)
    assert data_set.get_dataset_size() == 3


def test_cv_minddataset_repeat_reshuffle(add_and_remove_cv_file):
    """tutorial for cv minddataset."""
    columns_list = ["data", "label"]
    num_readers = 4
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers)
    decode_op = vision.Decode()
    data_set = data_set.map(input_columns=["data"], operations=decode_op, num_parallel_workers=2)
    resize_op = vision.Resize((32, 32), interpolation=Inter.LINEAR)
    data_set = data_set.map(input_columns="data", operations=resize_op, num_parallel_workers=2)
    data_set = data_set.batch(2)
    data_set = data_set.repeat(2)
    num_iter = 0
    labels = []
    for item in data_set.create_dict_iterator():
        logger.info("-------------- get dataset size {} -----------------".format(num_iter))
        logger.info("-------------- item[label]: {} ---------------------".format(item["label"]))
        logger.info("-------------- item[data]: {} ----------------------".format(item["data"]))
        num_iter += 1
        labels.append(item["label"])
    assert num_iter == 10
    logger.info("repeat shuffle: {}".format(labels))
    assert len(labels) == 10
    assert labels[0:5] == labels[0:5]
    assert labels[0:5] != labels[5:5]


def test_cv_minddataset_batch_size_larger_than_records(add_and_remove_cv_file):
    """tutorial for cv minddataset."""
    columns_list = ["data", "label"]
    num_readers = 4
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers)
    decode_op = vision.Decode()
    data_set = data_set.map(input_columns=["data"], operations=decode_op, num_parallel_workers=2)
    resize_op = vision.Resize((32, 32), interpolation=Inter.LINEAR)
    data_set = data_set.map(input_columns="data", operations=resize_op, num_parallel_workers=2)
    data_set = data_set.batch(32, drop_remainder=True)
    num_iter = 0
    for item in data_set.create_dict_iterator():
        logger.info("-------------- get dataset size {} -----------------".format(num_iter))
        logger.info("-------------- item[label]: {} ---------------------".format(item["label"]))
        logger.info("-------------- item[data]: {} ----------------------".format(item["data"]))
        num_iter += 1
    assert num_iter == 0


def test_cv_minddataset_issue_888(add_and_remove_cv_file):
    """issue 888 test."""
    columns_list = ["data", "label"]
    num_readers = 2
    data = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers, shuffle=False, num_shards=5, shard_id=1)
    data = data.shuffle(2)
    data = data.repeat(9)
    num_iter = 0
    for item in data.create_dict_iterator():
        num_iter += 1
    assert num_iter == 18


def test_cv_minddataset_blockreader_tutorial(add_and_remove_cv_file):
    """tutorial for cv minddataset."""
    columns_list = ["data", "label"]
    num_readers = 4
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
                              block_reader=True)
    assert data_set.get_dataset_size() == 10
    repeat_num = 2
    data_set = data_set.repeat(repeat_num)
    num_iter = 0
    for item in data_set.create_dict_iterator():
        logger.info("-------------- block reader repeat tow {} -----------------".format(num_iter))
        logger.info("-------------- item[label]: {} ----------------------------".format(item["label"]))
        logger.info("-------------- item[data]: {} -----------------------------".format(item["data"]))
        num_iter += 1
    assert num_iter == 20


def test_cv_minddataset_reader_basic_tutorial(add_and_remove_cv_file):
    """tutorial for cv minderdataset."""
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers)
    assert data_set.get_dataset_size() == 10
    num_iter = 0
    for item in data_set.create_dict_iterator():
        logger.info("-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info("-------------- len(item[data]): {} ------------------------".format(len(item["data"])))
        logger.info("-------------- item[data]: {} -----------------------------".format(item["data"]))
        logger.info("-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info("-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1
    assert num_iter == 10

def test_nlp_minddataset_reader_basic_tutorial(add_and_remove_nlp_file):
    """tutorial for nlp minderdataset."""
    num_readers = 4
    data_set = ds.MindDataset(NLP_FILE_NAME + "0", None, num_readers)
    assert data_set.get_dataset_size() == 10
    num_iter = 0
    for item in data_set.create_dict_iterator():
        logger.info("-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info("-------------- num_iter: {} ------------------------".format(num_iter))
        logger.info("-------------- item[id]: {} ------------------------".format(item["id"]))
        logger.info("-------------- item[rating]: {} --------------------".format(item["rating"]))
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
    """tutorial for cv minderdataset."""
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers)
    assert data_set.get_dataset_size() == 10
    for epoch in range(5):
        num_iter = 0
        for data in data_set:
            logger.info("data is {}".format(data))
            num_iter += 1
        assert num_iter == 10

        data_set.reset()


def test_cv_minddataset_reader_basic_tutorial_5_epoch_with_batch(add_and_remove_cv_file):
    """tutorial for cv minderdataset."""
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers)

    resize_height = 32
    resize_width = 32

    # define map operations
    decode_op = vision.Decode()
    resize_op = vision.Resize((resize_height, resize_width), ds.transforms.vision.Inter.LINEAR)

    data_set = data_set.map(input_columns=["data"], operations=decode_op, num_parallel_workers=4)
    data_set = data_set.map(input_columns=["data"], operations=resize_op, num_parallel_workers=4)

    data_set = data_set.batch(2)
    assert data_set.get_dataset_size() == 5
    for epoch in range(5):
        num_iter = 0
        for data in data_set:
            logger.info("data is {}".format(data))
            num_iter += 1
        assert num_iter == 5

        data_set.reset()


def test_cv_minddataset_reader_no_columns(add_and_remove_cv_file):
    """tutorial for cv minderdataset."""
    data_set = ds.MindDataset(CV_FILE_NAME + "0")
    assert data_set.get_dataset_size() == 10
    num_iter = 0
    for item in data_set.create_dict_iterator():
        logger.info("-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info("-------------- len(item[data]): {} ------------------------".format(len(item["data"])))
        logger.info("-------------- item[data]: {} -----------------------------".format(item["data"]))
        logger.info("-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info("-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1
    assert num_iter == 10


def test_cv_minddataset_reader_repeat_tutorial(add_and_remove_cv_file):
    """tutorial for cv minderdataset."""
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers)
    repeat_num = 2
    data_set = data_set.repeat(repeat_num)
    num_iter = 0
    for item in data_set.create_dict_iterator():
        logger.info("-------------- repeat two test {} ------------------------".format(num_iter))
        logger.info("-------------- len(item[data]): {} -----------------------".format(len(item["data"])))
        logger.info("-------------- item[data]: {} ----------------------------".format(item["data"]))
        logger.info("-------------- item[file_name]: {} -----------------------".format(item["file_name"]))
        logger.info("-------------- item[label]: {} ---------------------------".format(item["label"]))
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
    for line in lines:
        try:
            filename, label = line.split(",")
            label = label.strip("\n")
            with open(os.path.join(img_dir, filename), "rb") as file_reader:
                img = file_reader.read()
            data_json = {"file_name": filename,
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
    logger.info('{} images are missing'.format(len(file_list)-len(data_list)))
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
        return vectors[0:maxlen], [1]*maxlen, [0]*maxlen
    input_ = vectors+[0]*(maxlen-length)
    mask = [1]*length + [0]*(maxlen-length)
    segment = [0]*maxlen
    return input_, mask, segment
