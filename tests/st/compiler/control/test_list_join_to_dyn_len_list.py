# Copyright 2023 Huawei Technologies Co., Ltd
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

import mindspore as ms


def test_value_list_join():
    """
    Feature: Abstract join.
    Description: Enable different size list join as dynamic len list.
    Expectation: No exception raised and the result is correct.
    """

    @ms.jit
    def net(x):
        list1 = [1, 2]
        list2 = [3, 4, 5]
        if x > 1:
            list3 = list1
        else:
            list3 = list2
        out = list3[1] + list3[0]
        return out

    out = net(ms.Tensor([2]))
    assert out == 3
