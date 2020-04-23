#!/usr/bin/env python
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

import os

import pytest

import mindspore.dataset as ds
from mindspore.mindrecord import FileWriter

CV_FILE_NAME = "./imagenet.mindrecord"


def create_cv_mindrecord(files_num):
    """tutorial for cv dataset writer."""
    os.remove(CV_FILE_NAME) if os.path.exists(CV_FILE_NAME) else None
    os.remove("{}.db".format(CV_FILE_NAME)) if os.path.exists("{}.db".format(CV_FILE_NAME)) else None
    writer = FileWriter(CV_FILE_NAME, files_num)
    cv_schema_json = {"file_name": {"type": "string"}, "label": {"type": "int32"}, "data": {"type": "bytes"}}
    data = [{"file_name": "001.jpg", "label": 43, "data": bytes('0xffsafdafda', encoding='utf-8')}]
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    writer.write_raw_data(data)
    writer.commit()


def test_cv_lack_json():
    """tutorial for cv minderdataset."""
    create_cv_mindrecord(1)
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    with pytest.raises(Exception) as err:
        data_set = ds.MindDataset(CV_FILE_NAME, "no_exist.json", columns_list, num_readers)
    os.remove(CV_FILE_NAME)
    os.remove("{}.db".format(CV_FILE_NAME))


def test_cv_lack_mindrecord():
    """tutorial for cv minderdataset."""
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    with pytest.raises(Exception, match="does not exist or permission denied"):
        data_set = ds.MindDataset("no_exist.mindrecord", columns_list, num_readers)


def test_invalid_mindrecord():
    with open('dummy.mindrecord', 'w') as f:
        f.write('just for test')
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    with pytest.raises(Exception, match="MindRecordOp init failed"):
        data_set = ds.MindDataset('dummy.mindrecord', columns_list, num_readers)
        num_iter = 0
        for item in data_set.create_dict_iterator():
            num_iter += 1
        assert num_iter == 0
    os.remove('dummy.mindrecord')


def test_minddataset_lack_db():
    create_cv_mindrecord(1)
    os.remove("{}.db".format(CV_FILE_NAME))
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    with pytest.raises(Exception, match="MindRecordOp init failed"):
        data_set = ds.MindDataset(CV_FILE_NAME, columns_list, num_readers)
        num_iter = 0
        for item in data_set.create_dict_iterator():
            num_iter += 1
        assert num_iter == 0
    os.remove(CV_FILE_NAME)


def test_cv_minddataset_pk_sample_error_class_column():
    create_cv_mindrecord(1)
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.PKSampler(5, None, True, 'no_exsit_column')
    with pytest.raises(Exception, match="MindRecordOp launch failed"):
        data_set = ds.MindDataset(CV_FILE_NAME, columns_list, num_readers, sampler=sampler)
        num_iter = 0
        for item in data_set.create_dict_iterator():
            num_iter += 1
    os.remove(CV_FILE_NAME)
    os.remove("{}.db".format(CV_FILE_NAME))

