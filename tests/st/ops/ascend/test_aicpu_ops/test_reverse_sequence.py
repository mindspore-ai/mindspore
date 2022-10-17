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
from mindspore.common.api import ms_function
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self, seq_dim, batch_dim):
        super(Net, self).__init__()
        self.reverse_sequence = P.ReverseSequence(seq_dim=seq_dim, batch_dim=batch_dim)

    @ms_function
    def construct(self, x, seq_lengths):
        return self.reverse_sequence(x, seq_lengths)


def test_net_int8():
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.int8)
    seq_lengths = np.array([1, 2, 3]).astype(np.int32)
    seq_dim = 0
    batch_dim = 1
    net = Net(seq_dim, batch_dim)
    output = net(Tensor(x), Tensor(seq_lengths))
    expected = np.array([[1, 5, 9], [4, 2, 6], [7, 8, 3]]).astype(np.int8)
    assert np.array_equal(output.asnumpy(), expected)


def test_net_int32():
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.int32)
    seq_lengths = np.array([1, 2, 3]).astype(np.int64)
    seq_dim = 1
    batch_dim = 0
    net = Net(seq_dim, batch_dim)
    output = net(Tensor(x), Tensor(seq_lengths))
    expected = np.array([[1, 2, 3], [5, 4, 6], [9, 8, 7]]).astype(np.int32)
    assert np.array_equal(output.asnumpy(), expected)


def test_reverse_sequence_tensor_api():
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mstype.int8)
    seq_lengths = Tensor(np.array([1, 2, 3]))
    output = x.reverse_sequence(seq_lengths, seq_dim=1)
    expected = np.array([[1, 2, 3], [5, 4, 6], [9, 8, 7]]).astype(np.int8)
    assert np.array_equal(output.asnumpy(), expected)
