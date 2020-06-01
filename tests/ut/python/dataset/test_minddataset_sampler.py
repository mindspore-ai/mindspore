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
import os
import pytest

import mindspore.dataset as ds
from mindspore import log as logger
from mindspore.dataset.text import to_str
from mindspore.mindrecord import FileWriter

FILES_NUM = 4
CV_FILE_NAME = "../data/mindrecord/imagenet.mindrecord"
CV_DIR_NAME = "../data/mindrecord/testImageNetData"


@pytest.fixture
def add_and_remove_cv_file():
    """add/remove cv file"""
    paths = ["{}{}".format(CV_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    for x in paths:
        if os.path.exists("{}".format(x)):
            os.remove("{}".format(x))
        if os.path.exists("{}.db".format(x)):
            os.remove("{}.db".format(x))
    writer = FileWriter(CV_FILE_NAME, FILES_NUM)
    data = get_data(CV_DIR_NAME, True)
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


def test_cv_minddataset_pk_sample_no_column(add_and_remove_cv_file):
    """tutorial for cv minderdataset."""
    num_readers = 4
    sampler = ds.PKSampler(2)
    data_set = ds.MindDataset(CV_FILE_NAME + "0", None, num_readers,
                              sampler=sampler)

    assert data_set.get_dataset_size() == 6
    num_iter = 0
    for item in data_set.create_dict_iterator():
        logger.info("-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info("-------------- item[file_name]: \
                {}------------------------".format(to_str(item["file_name"])))
        logger.info("-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1


def test_cv_minddataset_pk_sample_basic(add_and_remove_cv_file):
    """tutorial for cv minderdataset."""
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.PKSampler(2)
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
                              sampler=sampler)

    assert data_set.get_dataset_size() == 6
    num_iter = 0
    for item in data_set.create_dict_iterator():
        logger.info("-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info("-------------- item[data]: \
                {}------------------------".format(item["data"][:10]))
        logger.info("-------------- item[file_name]: \
                {}------------------------".format(to_str(item["file_name"])))
        logger.info("-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1


def test_cv_minddataset_pk_sample_shuffle(add_and_remove_cv_file):
    """tutorial for cv minderdataset."""
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.PKSampler(3, None, True)
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
                              sampler=sampler)

    assert data_set.get_dataset_size() == 9
    num_iter = 0
    for item in data_set.create_dict_iterator():
        logger.info("-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info("-------------- item[file_name]: \
                {}------------------------".format(to_str(item["file_name"])))
        logger.info("-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1


def test_cv_minddataset_pk_sample_out_of_range(add_and_remove_cv_file):
    """tutorial for cv minderdataset."""
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.PKSampler(5, None, True)
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
                              sampler=sampler)
    assert data_set.get_dataset_size() == 15
    num_iter = 0
    for item in data_set.create_dict_iterator():
        logger.info("-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info("-------------- item[file_name]: \
                {}------------------------".format(to_str(item["file_name"])))
        logger.info("-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1


def test_cv_minddataset_subset_random_sample_basic(add_and_remove_cv_file):
    """tutorial for cv minderdataset."""
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    indices = [1, 2, 3, 5, 7]
    sampler = ds.SubsetRandomSampler(indices)
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
                              sampler=sampler)
    assert data_set.get_dataset_size() == 5
    num_iter = 0
    for item in data_set.create_dict_iterator():
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info(
            "-------------- item[data]: {}  -----------------------------".format(item["data"]))
        logger.info(
            "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1
    assert num_iter == 5


def test_cv_minddataset_subset_random_sample_replica(add_and_remove_cv_file):
    """tutorial for cv minderdataset."""
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    indices = [1, 2, 2, 5, 7, 9]
    sampler = ds.SubsetRandomSampler(indices)
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
                              sampler=sampler)
    assert data_set.get_dataset_size() == 6
    num_iter = 0
    for item in data_set.create_dict_iterator():
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info(
            "-------------- item[data]: {}  -----------------------------".format(item["data"]))
        logger.info(
            "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1
    assert num_iter == 6


def test_cv_minddataset_subset_random_sample_empty(add_and_remove_cv_file):
    """tutorial for cv minderdataset."""
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    indices = []
    sampler = ds.SubsetRandomSampler(indices)
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
                              sampler=sampler)
    assert data_set.get_dataset_size() == 0
    num_iter = 0
    for item in data_set.create_dict_iterator():
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info(
            "-------------- item[data]: {}  -----------------------------".format(item["data"]))
        logger.info(
            "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1
    assert num_iter == 0


def test_cv_minddataset_subset_random_sample_out_of_range(add_and_remove_cv_file):
    """tutorial for cv minderdataset."""
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    indices = [1, 2, 4, 11, 13]
    sampler = ds.SubsetRandomSampler(indices)
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
                              sampler=sampler)
    assert data_set.get_dataset_size() == 5
    num_iter = 0
    for item in data_set.create_dict_iterator():
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info(
            "-------------- item[data]: {}  -----------------------------".format(item["data"]))
        logger.info(
            "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1
    assert num_iter == 5


def test_cv_minddataset_subset_random_sample_negative(add_and_remove_cv_file):
    """tutorial for cv minderdataset."""
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    indices = [1, 2, 4, -1, -2]
    sampler = ds.SubsetRandomSampler(indices)
    data_set = ds.MindDataset(CV_FILE_NAME + "0", columns_list, num_readers,
                              sampler=sampler)
    assert data_set.get_dataset_size() == 5
    num_iter = 0
    for item in data_set.create_dict_iterator():
        logger.info(
            "-------------- cv reader basic: {} ------------------------".format(num_iter))
        logger.info(
            "-------------- item[data]: {}  -----------------------------".format(item["data"]))
        logger.info(
            "-------------- item[file_name]: {} ------------------------".format(item["file_name"]))
        logger.info(
            "-------------- item[label]: {} ----------------------------".format(item["label"]))
        num_iter += 1
    assert num_iter == 5


def get_data(dir_name, sampler=False):
    """
    usage: get data from imagenet dataset
    params:
    dir_name: directory containing folder images and annotation information

    """
    if not os.path.isdir(dir_name):
        raise IOError("Directory {} not exists".format(dir_name))
    img_dir = os.path.join(dir_name, "images")
    if sampler:
        ann_file = os.path.join(dir_name, "annotation_sampler.txt")
    else:
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
