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
""" multi_relu_case """
from mindspore.ops import functional as F


# Test user define ops
def get_test_ops_fn():
    return test_ops_f


scalar_mul = F.scalar_mul


def test_ops_f(x, y):
    z = scalar_mul(x, y)
    return z
