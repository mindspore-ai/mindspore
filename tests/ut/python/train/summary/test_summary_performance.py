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
@File  : test_summary.py
@Author:
@Date  : 2019-07-4
@Desc  : test summary function
"""
import os
import logging
import time
import numpy as np
from mindspore.train.summary.summary_record import SummaryRecord, _cache_summary_tensor_data
from mindspore.common.tensor import Tensor

CUR_DIR = os.getcwd()
SUMMARY_DIR = CUR_DIR + "/test_temp_summary_event_file/"

log = logging.getLogger("test")
log.setLevel(level=logging.ERROR)

def get_now_time_ns():
    """get the time of second"""
    time_second = int(time.time_ns())
    return time_second

def get_test_data(step):
    """ get_test_data """
    # pylint: disable=unused-argument
    test_data_list = []
    tag1 = "xt1[:Tensor]"
    tag2 = "xt2[:Tensor]"
    tag3 = "xt3[:Tensor]"
    np1 = np.random.random((50, 40, 30, 50))
    np2 = np.random.random((50, 50, 30, 50))
    np3 = np.random.random((40, 55, 30, 50))

    dict1 = {}
    dict1["name"] = tag1
    dict1["data"] = Tensor(np1)

    dict2 = {}
    dict2["name"] = tag2
    dict2["data"] = Tensor(np2)

    dict3 = {}
    dict3["name"] = tag3
    dict3["data"] = Tensor(np3)

    test_data_list.append(dict1)
    test_data_list.append(dict2)

    return test_data_list


# Test 1: summary sample of scalar
def test_summary_performance():
    """ test_summary_performance """
    log.debug("begin test_scalar_summary_sample")
    current_time = time.time()
    print("time = ", current_time)
    # step 0: create the thread
    test_writer = SummaryRecord(SUMMARY_DIR, flush_time=120)

    # step 1: create the test data for summary
    old_time = get_now_time_ns()
    # step 2: create the Event
    for i in range(1, 10):
        test_data = get_test_data(i)
        _cache_summary_tensor_data(test_data)
        test_writer.record(i)
        now_time = get_now_time_ns()
        consume_time = (now_time - old_time)/1000/1000
        old_time = now_time
        print("step test_summary_performance conusmer time is:", consume_time)


    # step 3: send the event to mq

    # step 4: accept the event and write the file
    test_writer.flush()
    test_writer.close()
    current_time = time.time() - current_time
    print("consume time = ", current_time)
    log.debug("finished test_scalar_summary_sample")
