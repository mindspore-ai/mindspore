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

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.ctc = P.CTCGreedyDecoder()

    def construct(self, inputs, sequence_length):
        return self.ctc(inputs, sequence_length)


def test_net_float32():
    x = np.random.randn(2, 2, 3).astype(np.float32)
    sequence_length = np.array([2, 2]).astype(np.int32)
    net = Net()
    output = net(Tensor(x), Tensor(sequence_length))
    print(output)


def test_net_assert():
    x = np.array([[[0.44662005, 0.41900548, -0.8334965],
                   [-0.28560895, -0.03626213, -0.04149306]],
                  [[-0.70390207, 0.2977548, -0.4097819],
                   [-0.6942656, -0.14625494, -0.90554816]]]).astype(np.float32)
    sequence_length = np.array([2, 2]).astype(np.int32)
    net = Net()
    output = net(Tensor(x), Tensor(sequence_length))
    print(output)

    out_expect0 = np.array([0, 0, 0, 1, 1, 0]).reshape(3, 2)
    out_expect1 = np.array([0, 1, 1])
    out_expect2 = np.array([2, 2])
    out_expect3 = np.array([-0.7443749, 0.18251707]).astype(np.float32).reshape(2, 1)
    assert np.array_equal(output[0].asnumpy(), out_expect0)
    assert np.array_equal(output[1].asnumpy(), out_expect1)
    assert np.array_equal(output[2].asnumpy(), out_expect2)
    assert np.array_equal(output[3].asnumpy(), out_expect3)
