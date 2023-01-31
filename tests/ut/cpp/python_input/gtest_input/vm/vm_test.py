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
""" vm_test """
from mindspore.ops import functional as F

scala_add = F.scalar_add
scala_mul = F.scalar_mul
scalar_gt = F.scalar_gt


def ScalarAdd(x, y):
    """Implement `scalar_add`."""
    return scala_add(x, y)


def ScalarMul(x, y):
    """Implement `scalar_mul`."""
    return scala_mul(x, y)


def test_if(x, y):
    if scalar_gt(x, y):
        return x
    return y
