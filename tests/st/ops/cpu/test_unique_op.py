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
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.uniq = P.Unique()

    def construct(self, x):
        return self.uniq(x)


def test_net_fp32():
    x = Tensor(np.array([1, 2, 5, 2]), mstype.float32)
    uniq = Net()
    output = uniq(x)
    print("x:\n", x)
    print("y:\n", output[0])
    print("idx:\n", output[1])
    expect_y_result = [1., 2., 5.]
    expect_idx_result = [0, 1, 2, 1]

    assert (output[0].asnumpy() == expect_y_result).all()
    assert (output[1].asnumpy() == expect_idx_result).all()


def test_net_fp16():
    x = Tensor(np.array([1, 5, 2, 2]), mstype.float16)
    uniq = Net()
    output = uniq(x)
    print("x:\n", x)
    print("y:\n", output[0])
    print("idx:\n", output[1])
    expect_y_result = [1., 5., 2.]
    expect_idx_result = [0, 1, 2, 2]

    assert (output[0].asnumpy() == expect_y_result).all()
    assert (output[1].asnumpy() == expect_idx_result).all()


def test_net_int32():
    x = Tensor(np.array([1, 2, 5, 2]), mstype.int32)
    uniq = Net()
    output = uniq(x)
    print("x:\n", x)
    print("y:\n", output[0])
    print("idx:\n", output[1])
    expect_y_result = [1, 2, 5]
    expect_idx_result = [0, 1, 2, 1]

    assert (output[0].asnumpy() == expect_y_result).all()
    assert (output[1].asnumpy() == expect_idx_result).all()


def test_net_int64():
    x = Tensor(np.array([1, 2, 5, 2]), mstype.int64)
    uniq = Net()
    output = uniq(x)
    print("x:\n", x)
    print("y:\n", output[0])
    print("idx:\n", output[1])
    expect_y_result = [1, 2, 5]
    expect_idx_result = [0, 1, 2, 1]

    assert (output[0].asnumpy() == expect_y_result).all()
    assert (output[1].asnumpy() == expect_idx_result).all()
