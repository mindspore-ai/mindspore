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
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE,
                    device_target="Ascend")


class Net(nn.Cell):
    def __init__(self, pad_num):
        super(Net, self).__init__()
        self.unique_with_pad = P.UniqueWithPad()
        self.pad_num = pad_num

    def construct(self, x):
        return self.unique_with_pad(x, self.pad_num)


def test_unique_with_pad():
    x = Tensor(np.array([1, 1, 5, 5, 4, 4, 3, 3, 2, 2]), mstype.int32)
    pad_num = 8
    unique_with_pad = Net(pad_num)
    out = unique_with_pad(x)
    expect_val = ([1, 5, 4, 3, 2, 8, 8, 8, 8, 8], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    assert(out[0].asnumpy() == expect_val[0]).all()
    assert(out[1].asnumpy() == expect_val[1]).all()
