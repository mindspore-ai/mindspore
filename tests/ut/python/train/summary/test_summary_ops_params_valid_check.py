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
@Date  : 2019-08-5
@Desc  : test summary function of ops params valid check
"""
import os
import logging
import random
import numpy as np
from mindspore.train.summary.summary_record import SummaryRecord
from mindspore.common.tensor import Tensor
import mindspore.nn as nn
from mindspore.ops import operations as P


CUR_DIR = os.getcwd()
SUMMARY_DIR = CUR_DIR + "/test_temp_summary_event_file/"

log = logging.getLogger("test")
log.setLevel(level=logging.ERROR)


class SummaryDemoTag(nn.Cell):
    """ SummaryDemoTag definition """
    def __init__(self, tag1, tag2, tag3):
        super(SummaryDemoTag, self).__init__()
        self.s = P.ScalarSummary()
        self.add = P.TensorAdd()
        self.tag1 = tag1
        self.tag2 = tag2
        self.tag3 = tag3

    def construct(self, x, y):
        self.s(self.tag1, x)
        z = self.add(x, y)
        self.s(self.tag2, z)
        self.s(self.tag3, y)
        return z


class SummaryDemoTagForSet(nn.Cell):
    """ SummaryDemoTagForSet definition """
    def __init__(self, tag_tuple):
        super(SummaryDemoTagForSet, self).__init__()
        self.s = P.ScalarSummary()
        self.add = P.TensorAdd()
        self.tag_tuple = tag_tuple

    def construct(self, x, y):
        z = self.add(x, y)
        for tag in self.tag_tuple:
            self.s(tag, x)
        return z


class SummaryDemoValue(nn.Cell):
    """ SummaryDemoValue definition """
    def __init__(self, value):
        super(SummaryDemoValue, self).__init__()
        self.s = P.ScalarSummary()
        self.add = P.TensorAdd()
        self.v = value

    def construct(self, x, y):
        self.s("x", self.v)
        z = self.add(x, y)
        self.s("z", self.v)
        self.s("y", self.v)
        return z

class SummaryDemoValueForSet(nn.Cell):
    """ SummaryDemoValueForSet definition """
    def __init__(self, value, tag_tuple):
        super(SummaryDemoValueForSet, self).__init__()
        self.s = P.ScalarSummary()
        self.add = P.TensorAdd()
        self.tag_tuple = tag_tuple
        self.v = value

    def construct(self, x, y):
        z = self.add(x, y)
        for tag in self.tag_tuple:
            self.s(tag, self.v)
        return z

def run_case(net):
    """ run_case """
    # step 0: create the thread
    test_writer = SummaryRecord(SUMMARY_DIR)

    # step 1: create the network for summary
    x = Tensor(np.array([1.1]).astype(np.float32))
    y = Tensor(np.array([1.2]).astype(np.float32))
    net.set_train()

    # step 2: create the Event
    steps = 100
    for i in range(1, steps):
        x = Tensor(np.array([1.1 + random.uniform(1, 10)]).astype(np.float32))
        y = Tensor(np.array([1.2 + random.uniform(1, 10)]).astype(np.float32))
        net(x, y)
        test_writer.record(i)

    # step 3: close the writer
    test_writer.close()


# Test 1: use the repeat tag
def test_scalar_summary_use_repeat_tag():
    log.debug("begin test_scalar_summary_use_repeat_tag")
    net = SummaryDemoTag("x", "x", "x")
    try:
        run_case(net)
    except:
        assert False
    else:
        assert True
    log.debug("finished test_scalar_summary_use_repeat_tag")


# Test 2: repeat tag use for set summary
def test_scalar_summary_use_repeat_tag_for_set():
    log.debug("begin test_scalar_summary_use_repeat_tag_for_set")
    net = SummaryDemoTagForSet(("x", "x", "x"))
    try:
        run_case(net)
    except:
        assert False
    else:
        assert True
    log.debug("finished test_scalar_summary_use_repeat_tag_for_set")


# Test3: test with invalid tag(None, bool, "", int)
def test_scalar_summary_use_invalid_tag_None():
    log.debug("begin test_scalar_summary_use_invalid_tag_None")
    net = SummaryDemoTag(None, None, None)
    try:
        run_case(net)
    except:
        assert True
    else:
        assert False
    log.debug("finished test_scalar_summary_use_invalid_tag_None")


# Test4: test with invalid tag(None, bool, "", int)
def test_scalar_summary_use_invalid_tag_Bool():
    log.debug("begin test_scalar_summary_use_invalid_tag_Bool")
    net = SummaryDemoTag(True, True, True)
    try:
        run_case(net)
    except:
        assert True
    else:
        assert False
    log.debug("finished test_scalar_summary_use_invalid_tag_Bool")


# Test5: test with invalid tag(None, bool, "", int)
def test_scalar_summary_use_invalid_tag_null():
    log.debug("begin test_scalar_summary_use_invalid_tag_null")
    net = SummaryDemoTag("", "", "")
    try:
        run_case(net)
    except:
        assert True
    else:
        assert False
    log.debug("finished test_scalar_summary_use_invalid_tag_null")


# Test6: test with invalid tag(None, bool, "", int)
def test_scalar_summary_use_invalid_tag_Int():
    log.debug("begin test_scalar_summary_use_invalid_tag_Int")
    net = SummaryDemoTag(1, 2, 3)
    try:
        run_case(net)
    except:
        assert True
    else:
        assert False
    log.debug("finished test_scalar_summary_use_invalid_tag_Int")


# Test7: test with invalid value(None, "")
def test_scalar_summary_use_invalid_value_None():
    log.debug("begin test_scalar_summary_use_invalid_tag_Int")
    net = SummaryDemoValue(None)
    try:
        run_case(net)
    except:
        assert True
    else:
        assert False
    log.debug("finished test_scalar_summary_use_invalid_tag_Int")



# Test8: test with invalid value(None, "")
def test_scalar_summary_use_invalid_value_None_ForSet():
    log.debug("begin test_scalar_summary_use_invalid_value_None_ForSet")
    try:
        net = SummaryDemoValueForSet(None, ("x1", "x2"))
        run_case(net)
    except:
        assert True
    else:
        assert False
    log.debug("finished test_scalar_summary_use_invalid_value_None_ForSet")


# Test9: test with invalid value(None, "")
def test_scalar_summary_use_invalid_value_null():
    log.debug("begin test_scalar_summary_use_invalid_value_null")
    try:
        net = SummaryDemoValue("")
        run_case(net)
    except:
        assert True
    else:
        assert False
    log.debug("finished test_scalar_summary_use_invalid_value_null")
