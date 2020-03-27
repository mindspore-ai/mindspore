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
from mindspore.ops import Primitive

scala_add = Primitive('scalar_add')
scala_mul = Primitive('scalar_mul')
def scalar_add(x, y):
    """Implement `scalar_add`."""
    return scala_add(x, y)

def scalar_mul(x, y):
    """Implement `scalar_mul`."""
    return scala_mul(x, y)

def test_if(x, y):
    if x > y:
        return x
    return y
