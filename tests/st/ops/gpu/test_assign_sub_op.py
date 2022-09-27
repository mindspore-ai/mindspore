# Copyright 2019 Huawei Technologies Co., Ltd
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
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P


class AssignSub(nn.Cell):
    def __init__(self, value):
        super(AssignSub, self).__init__()
        self.var = Parameter(value, name="var")
        self.sub = P.AssignSub()

    def construct(self, y):
        res = self.sub(self.var, y)
        return res


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_assign_sub():
    """
    Feature: assign sub kernel
    Description: test assignsub
    Expectation: just test
    """
    expect1 = np.zeros([1, 3, 3, 3])
    expect2 = np.array([[[[0, -1, -2],
                          [-3, -4, -5],
                          [-6, -7, -8]],
                         [[-9, -10, -11],
                          [-12, -13, -14],
                          [-15, -16, -17]],
                         [[-18, -19, -20],
                          [-21, -22, -23],
                          [-24, -25, -26]]]])

    x1 = Tensor(np.arange(1 * 3 * 3 * 3).reshape(1, 3, 3, 3).astype(np.float32))
    y1 = Tensor(np.arange(1 * 3 * 3 * 3).reshape(1, 3, 3, 3).astype(np.float32))

    x2 = Tensor(np.arange(1 * 3 * 3 * 3).reshape(1, 3, 3, 3).astype(np.float32))
    y2 = Tensor(np.arange(1 * 3 * 3 * 3).reshape(1, 3, 3, 3).astype(np.float32))

    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    sub = AssignSub(x1)
    output1 = sub(y1)
    assert (output1.asnumpy() == expect1).all()
    sub = AssignSub(output1)
    output2 = sub(y1)
    assert (output2.asnumpy() == expect2).all()

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    sub = AssignSub(x2)
    output1 = sub(y2)
    assert (output1.asnumpy() == expect1).all()
    sub = AssignSub(output1)
    output2 = sub(y2)
    assert (output2.asnumpy() == expect2).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_assign_sub_float16():
    """
    Feature: None
    Description: test assignsub float16
    Expectation: just test
    """
    expect3 = np.zeros([1, 3, 3, 3])
    expect4 = np.array([[[[0, -1, -2],
                          [-3, -4, -5],
                          [-6, -7, -8]],
                         [[-9, -10, -11],
                          [-12, -13, -14],
                          [-15, -16, -17]],
                         [[-18, -19, -20],
                          [-21, -22, -23],
                          [-24, -25, -26]]]])

    x1 = Tensor(np.arange(1 * 3 * 3 * 3).reshape(1, 3, 3, 3).astype(np.float16))
    y1 = Tensor(np.arange(1 * 3 * 3 * 3).reshape(1, 3, 3, 3).astype(np.float16))

    x2 = Tensor(np.arange(1 * 3 * 3 * 3).reshape(1, 3, 3, 3).astype(np.float16))
    y2 = Tensor(np.arange(1 * 3 * 3 * 3).reshape(1, 3, 3, 3).astype(np.float16))

    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    sub = AssignSub(x1)
    output1 = sub(y1)
    assert (output1.asnumpy() == expect3).all()
    sub = AssignSub(output1)
    output2 = sub(y1)
    assert (output2.asnumpy() == expect4).all()

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    sub = AssignSub(x2)
    output1 = sub(y2)
    assert (output1.asnumpy() == expect3).all()
    sub = AssignSub(output1)
    output2 = sub(y2)
    assert (output2.asnumpy() == expect4).all()
