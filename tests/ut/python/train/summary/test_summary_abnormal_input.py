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
# ============================================================================
"""
@File  : test_summary_abnormal_input.py
@Author:
@Date  : 2019-08-5
@Desc  : test summary function of abnormal input
"""
import logging
import os
import numpy as np

from mindspore.common.tensor import Tensor
from mindspore.train.summary.summary_record import SummaryRecord

CUR_DIR = os.getcwd()
SUMMARY_DIR = CUR_DIR + "/test_temp_summary_event_file/"

log = logging.getLogger("test")
log.setLevel(level=logging.ERROR)


def get_test_data(step):
    """ get_test_data """
    test_data_list = []
    tag1 = "x1[:Scalar]"
    tag2 = "x2[:Scalar]"
    np1 = np.array(step + 1).astype(np.float32)
    np2 = np.array(step + 2).astype(np.float32)

    dict1 = {}
    dict1["name"] = tag1
    dict1["data"] = Tensor(np1)

    dict2 = {}
    dict2["name"] = tag2
    dict2["data"] = Tensor(np2)

    test_data_list.append(dict1)
    test_data_list.append(dict2)

    return test_data_list


# Test: call method on parse graph code
def test_summaryrecord_input_null_string():
    log.debug("begin test_summaryrecord_input_null_string")
    # step 0: create the thread
    try:
        SummaryRecord("")
    except:
        assert True
    else:
        assert False
    log.debug("finished test_summaryrecord_input_null_string")


def test_summaryrecord_input_None():
    log.debug("begin test_summaryrecord_input_None")
    # step 0: create the thread
    try:
        SummaryRecord(None)
    except:
        assert True
    else:
        assert False
    log.debug("finished test_summaryrecord_input_None")


def test_summaryrecord_input_relative_dir_1():
    log.debug("begin test_summaryrecord_input_relative_dir_1")
    # step 0: create the thread
    try:
        SummaryRecord("./test_temp_summary_event_file/")
    except:
        assert False
    else:
        assert True
    log.debug("finished test_summaryrecord_input_relative_dir_1")


def test_summaryrecord_input_relative_dir_2():
    log.debug("begin test_summaryrecord_input_relative_dir_2")
    # step 0: create the thread
    try:
        SummaryRecord("../summary/")
    except:
        assert False
    else:
        assert True
    log.debug("finished test_summaryrecord_input_relative_dir_2")


def test_summaryrecord_input_invalid_type_dir():
    log.debug("begin test_summaryrecord_input_invalid_type_dir")
    # step 0: create the thread
    try:
        SummaryRecord(32)
    except:
        assert True
    else:
        assert False
    log.debug("finished test_summaryrecord_input_invalid_type_dir")


def test_mulit_layer_directory():
    log.debug("begin test_mulit_layer_directory")
    # step 0: create the thread
    try:
        SummaryRecord("./test_temp_summary_event_file/test/t1/")
    except:
        assert False
    else:
        assert True
    log.debug("finished test_mulit_layer_directory")
