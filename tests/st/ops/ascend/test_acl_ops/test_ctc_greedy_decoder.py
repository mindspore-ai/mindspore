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

import numpy as np

import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.ops import operations as P
from tests.mark_utils import arg_mark

context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self, merge_repeated=True):
        super(Net, self).__init__()
        self.ctc = P.CTCGreedyDecoder(merge_repeated=merge_repeated)

    def construct(self, inputs, sequence_length):
        return self.ctc(inputs, sequence_length)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_ctc_greedy_deocder_float32():
    """
    Feature: CTCGreedyDecoder cpu op
    Description: Test output for fp32 dtype
    Expectation: Output matching expected values
    """
    inputs_np = np.array([[[1.7640524, 0.4001572, 0.978738],
                           [2.2408931, 1.867558, -0.9772779]],
                          [[0.95008844, -0.1513572, -0.10321885],
                           [0.41059852, 0.14404356, 1.4542735]]]).astype(np.float32)
    sequence_length_np = np.array([1, 1]).astype(np.int32)
    net = Net()
    output = net(Tensor(inputs_np), Tensor(sequence_length_np))

    out_expect0 = np.array([0, 0, 1, 0]).reshape(2, 2)
    out_expect1 = np.array([0, 0])
    out_expect2 = np.array([2, 1])
    out_expect3 = np.array([-1.7640524, -2.2408931]).astype(np.float32).reshape(2, 1)

    assert np.array_equal(output[0].asnumpy(), out_expect0)
    assert np.array_equal(output[1].asnumpy(), out_expect1)
    assert np.array_equal(output[2].asnumpy(), out_expect2)
    assert np.array_equal(output[3].asnumpy(), out_expect3)
