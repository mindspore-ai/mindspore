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
from mindspore.ops import operations as P

context.set_context(device_target="Ascend")


class Net(nn.Cell):
    def __init__(self, num_true=1):
        super(Net, self).__init__()
        self.sampler = P.ComputeAccidentalHits(num_true)

    def construct(self, x, y):
        return self.sampler(x, y)


def test_net_graph():
    """
    Feature: ComputeAccidentalHits is a dynamic operator on GRAPH mode
    Description:  Test operator ComputeAccidentalHits for geOP
    Expectation: the result of ComputeAccidentalHits is correct.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = np.array([[1, 2], [0, 4], [3, 3]])
    y = np.array([0, 1, 2, 3, 4])
    net = Net(2)
    output1, output2, output3 = net(Tensor(x), Tensor(y))
    print(output1, output2, output3)

    output1_expect = np.array([0, 0, 1, 1, 2, 2])
    output2_expect = np.array([1, 2, 0, 4, 3, 3])
    output3_expect = np.array([-3.4028235e+38, -3.4028235e+38, -3.4028235e+38,
                               -3.4028235e+38, -3.4028235e+38, -3.4028235e+38]).astype(np.float32)
    assert np.array_equal(output1.asnumpy(), output1_expect)
    assert np.array_equal(output2.asnumpy(), output2_expect)
    assert np.array_equal(output3.asnumpy(), output3_expect)


def test_net_pynative():
    """
    Feature: ComputeAccidentalHits is a dynamic operator on PyNative mode
    Description:   Test operator ComputeAccidentalHits for AclOP
    Expectation: the result of ComputeAccidentalHits is correct.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = np.array([[1, 0, 2], [1, 0, -2]])
    y = np.array([2, 4, -1, 5, 0])
    net = Net(3)
    output1, output2, output3 = net(Tensor(x), Tensor(y))
    print(output1, output2, output3)

    output1_expect = np.array([0, 0, 1])
    output2_expect = np.array([4, 0, 4])
    output3_expect = np.array([-3.4028235e+38, -3.4028235e+38, -3.4028235e+38]).astype(np.float32)
    assert np.array_equal(output1.asnumpy(), output1_expect)
    assert np.array_equal(output2.asnumpy(), output2_expect)
    assert np.array_equal(output3.asnumpy(), output3_expect)
