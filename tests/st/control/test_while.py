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
import numpy as np
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, ms_function
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P


@ms_function
def t1_while(x, y, z):
    y = y + 4
    while x < y:
        x = x + 1
    x = x + 3
    return x


def test_net():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    c1 = Tensor([2], mstype.int32)
    c2 = Tensor([14], mstype.int32)
    c3 = Tensor([1], mstype.int32)
    expect = Tensor([21], mstype.int32)
    ret = t1_while(c1, c2, c3)
    assert (ret == expect)


if __name__ == "__main__":
    test_net()
