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
@File  : parser_abnormal.py
@Author:
@Date  : 2019-03-14 18:37
@Desc  : parser test function.
"""


def nonrec():
    return 123


def rec1():
    return rec2()


def rec2():
    return rec1()


def test_keep_roots_recursion():
    return rec1() + nonrec()


def test_f(x, y):
    return x + y


def test_performance(x):
    return x


def test_list_append(x, y):
    list_test = []
    list_test.append(x)
    list_test.append(y)
    return list_test
