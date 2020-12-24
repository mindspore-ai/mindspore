# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""Test summary."""
import os
import random

import numpy as np


import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.train.summary.summary_record import SummaryRecord, _cache_summary_tensor_data

CUR_DIR = os.getcwd()
SUMMARY_DIR = CUR_DIR + "/test_temp_summary_event_file/"


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


def test_scalar_summary_sample():
    """ test_scalar_summary_sample """
    with SummaryRecord(SUMMARY_DIR, file_suffix="_MS_SCALAR") as test_writer:
        for i in range(1, 5):
            test_data = get_test_data(i)
            _cache_summary_tensor_data(test_data)
            test_writer.record(i)


def get_test_data_shape_1(step):
    """ get_test_data_shape_1 """
    test_data_list = []
    tag1 = "x1[:Scalar]"
    tag2 = "x2[:Scalar]"
    np1 = np.array([step + 1]).astype(np.float32)
    np2 = np.array([step + 2]).astype(np.float32)

    dict1 = {}
    dict1["name"] = tag1
    dict1["data"] = Tensor(np1)

    dict2 = {}
    dict2["name"] = tag2
    dict2["data"] = Tensor(np2)

    test_data_list.append(dict1)
    test_data_list.append(dict2)

    return test_data_list


# Test: shape = (1,)
def test_scalar_summary_sample_with_shape_1():
    """ test_scalar_summary_sample_with_shape_1 """
    with SummaryRecord(SUMMARY_DIR, file_suffix="_MS_SCALAR") as test_writer:
        for i in range(1, 100):
            test_data = get_test_data_shape_1(i)
            _cache_summary_tensor_data(test_data)
            test_writer.record(i)


# Test: test with ge
class SummaryDemo(nn.Cell):
    """ SummaryDemo definition """

    def __init__(self,):
        super(SummaryDemo, self).__init__()
        self.s = P.ScalarSummary()
        self.histogram_summary = P.HistogramSummary()
        self.add = P.TensorAdd()

    def construct(self, x, y):
        self.s("x1", x)
        z = self.add(x, y)
        self.s("z1", z)
        self.s("y1", y)
        self.histogram_summary("histogram", z)
        return z


def test_scalar_summary_with_ge():
    """ test_scalar_summary_with_ge """
    with SummaryRecord(SUMMARY_DIR, file_suffix="_MS_SCALAR") as test_writer:
        net = SummaryDemo()
        net.set_train()

        # step 2: create the Event
        steps = 100
        for i in range(1, steps):
            x = Tensor(np.array([1.1 + random.uniform(1, 10)]).astype(np.float32))
            y = Tensor(np.array([1.2 + random.uniform(1, 10)]).astype(np.float32))
            net(x, y)
            test_writer.record(i)


# test the problem of two consecutive use cases going wrong
def test_scalar_summary_with_ge_2():
    """ test_scalar_summary_with_ge_2 """
    with SummaryRecord(SUMMARY_DIR, file_suffix="_MS_SCALAR") as test_writer:
        net = SummaryDemo()
        net.set_train()

        steps = 100
        for i in range(1, steps):
            x = Tensor(np.array([1.1]).astype(np.float32))
            y = Tensor(np.array([1.2]).astype(np.float32))
            net(x, y)
            test_writer.record(i)
