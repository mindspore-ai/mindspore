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
import mindspore.dataset as ds
from mindspore import log as logger

FILES_NUM = 1
CV_FILE_NAME = "../data/mindrecord/testTwoImageData/twobytes.mindrecord"


def test_cv_minddataset_reader_two_png_tutorial():
    """
    Feature: MindDataset
    Description: Test for CV MindDataset basic reader with two pngs tutorial
    Expectation: Runs successfully
    """
    columns_list = ["id", "file_name", "label_name", "img_data", "label_data"]
    num_readers = 1
    data_set = ds.MindDataset(CV_FILE_NAME, columns_list, num_readers)
    assert data_set.get_dataset_size() == 5
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 5
        logger.info("-------------- cv reader basic is {} -----------------".format(num_iter))
        logger.info("-------------- item[id] is {} ------------------------".format(item["id"]))
        logger.info("-------------- item[file_name] is {} -----------------".format(item["file_name"]))
        logger.info("-------------- item[label_name] is {} ----------------".format(item["label_name"]))
        logger.info("-------------- item[img_data] is {} ------------------".format(item["img_data"]))
        logger.info("-------------- item[img_data][500:520] is {} ---------".format(item["img_data"][500:520]))
        logger.info("-------------- item[label_data] is {} ----------------".format(item["label_data"]))
        logger.info("-------------- item[label_data][500:520] is {} -------".format(item["label_data"][500:520]))
        num_iter += 1
    assert num_iter == 5


def test_cv_minddataset_reader_two_png_tutorial_just_image2():
    """
    Feature: MindDataset
    Description: Test for CV MindDataset basic reader with two pngs tutorial but only using image and label data
    Expectation: Runs successfully
    """
    columns_list = ["img_data", "label_data"]
    num_readers = 1
    data_set = ds.MindDataset(CV_FILE_NAME, columns_list, num_readers)
    assert data_set.get_dataset_size() == 5
    num_iter = 0
    for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 2
        logger.info("-------------- cv reader basic is {} -----------------".format(num_iter))
        logger.info("-------------- item[img_data] is {} ------------------".format(item["img_data"]))
        logger.info("-------------- item[img_data][500:520] is {} ---------".format(item["img_data"][500:520]))
        logger.info("-------------- item[label_data] is {} ----------------".format(item["label_data"]))
        logger.info("-------------- item[label_data][500:520] is {} -------".format(item["label_data"][500:520]))
        num_iter += 1
    assert num_iter == 5
