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
@File  : test_tensor_summary.py
@Author:
@Date  : 2019-07-4
@Desc  : test summary function
"""
import logging
import os

import numpy as np

import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.train.summary.summary_record import SummaryRecord, _cache_summary_tensor_data

CUR_DIR = os.getcwd()
SUMMARY_DIR = CUR_DIR + "/test_temp_summary_event_file/"

log = logging.getLogger("test")
log.setLevel(level=logging.ERROR)


def get_test_data(step):
    """ get_test_data """
    test_data_list = []

    dict_x1 = {}
    dict_x1["name"] = "x1[:Tensor]"
    dict_x1["data"] = Tensor(np.array([[1, 2, step + 1], [2, 3, 4]]).astype(np.int8))
    test_data_list.append(dict_x1)
    dict_x2 = {}
    dict_x2["name"] = "x2[:Tensor]"
    dict_x2["data"] = Tensor(np.array([[1, 2, step + 2], [2, 3, 4]]).astype(np.int16))
    test_data_list.append(dict_x2)
    dict_x3 = {}
    dict_x3["name"] = "x3[:Tensor]"
    dict_x3["data"] = Tensor(np.array([[1, 2, step + 1], [2, 3, 4]]).astype(np.int32))
    test_data_list.append(dict_x3)
    dict_x4 = {}
    dict_x4["name"] = "x4[:Tensor]"
    dict_x4["data"] = Tensor(np.array([[1, 2, step + 1], [2, 3, 4]]).astype(np.int64))
    test_data_list.append(dict_x4)
    dict_x5 = {}
    dict_x5["name"] = "x5[:Tensor]"
    dict_x5["data"] = Tensor(np.array([[1, 2, step + 1], [2, 3, 4]]).astype(np.float))
    test_data_list.append(dict_x5)
    dict_x6 = {}
    dict_x6["name"] = "x6[:Tensor]"
    dict_x6["data"] = Tensor(np.array([[1, 2, step + 1], [2, 3, 4]]).astype(np.float16))
    test_data_list.append(dict_x6)
    dict_x7 = {}
    dict_x7["name"] = "x7[:Tensor]"
    dict_x7["data"] = Tensor(np.array([[1, 2, step + 1], [2, 3, 4]]).astype(np.float32))
    test_data_list.append(dict_x7)
    dict_x8 = {}
    dict_x8["name"] = "x8[:Tensor]"
    dict_x8["data"] = Tensor(np.array([[1, 2, step + 1], [2, 3, 4]]).astype(np.float64))
    test_data_list.append(dict_x8)

    return test_data_list


# Test: call method on parse graph code
def test_tensor_summary_sample():
    """ test_tensor_summary_sample """
    log.debug("begin test_tensor_summary_sample")
    # step 0: create the thread
    with SummaryRecord(SUMMARY_DIR, file_suffix="_MS_TENSOR") as test_writer:

        # step 1: create the Event
        for i in range(1, 100):
            test_data = get_test_data(i)

            _cache_summary_tensor_data(test_data)
            test_writer.record(i)

        # step 2: accept the event and write the file

        log.debug("finished test_tensor_summary_sample")


def get_test_data_check(step):
    """ get_test_data_check """
    test_data_list = []
    tag1 = "x1[:Tensor]"
    np1 = np.array([[step, step, step], [2, 3, 4]]).astype(np.float32)

    dict1 = {}
    dict1["name"] = tag1
    dict1["data"] = Tensor(np1)
    test_data_list.append(dict1)

    return test_data_list


# Test: test with ge
class SummaryDemo(nn.Cell):
    """ SummaryDemo definition """

    def __init__(self, ):
        super(SummaryDemo, self).__init__()
        self.s = P.TensorSummary()
        self.add = P.TensorAdd()

    def construct(self, x, y):
        self.s("x1", x)
        z = self.add(x, y)
        self.s("z1", z)
        self.s("y1", y)
        return z


def test_tensor_summary_with_ge():
    """ test_tensor_summary_with_ge """
    log.debug("begin test_tensor_summary_with_ge")

    # step 0: create the thread
    with SummaryRecord(SUMMARY_DIR) as test_writer:

        # step 1: create the network for summary
        x = Tensor(np.array([1.1]).astype(np.float32))
        y = Tensor(np.array([1.2]).astype(np.float32))
        net = SummaryDemo()
        net.set_train()

        # step 2: create the Event
        steps = 100
        for i in range(1, steps):
            x = Tensor(np.array([[i], [i]]).astype(np.float32))
            y = Tensor(np.array([[i + 1], [i + 1]]).astype(np.float32))
            net(x, y)
            test_writer.record(i)

        log.debug("finished test_tensor_summary_with_ge")
