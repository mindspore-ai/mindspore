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
@File   : opt_clean.py
@Author : wangqiuliang
@Date   : 2019-03-18
@Desc   : parse python function for ut of erase class
"""
from dataclasses import dataclass


# Test_Erase_class
@dataclass
class Point:
    x: float
    y: float

    def product(self):
        return self.x * self.y


def test_erase_class_fn(p_in):
    p = Point(p_in)
    return p.x * p.y
