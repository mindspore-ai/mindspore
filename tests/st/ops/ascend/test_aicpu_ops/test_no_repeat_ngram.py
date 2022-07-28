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
import mindspore
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops import operations as P

FLT_MAX = 3.4028235e+38

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self, ngram_size):
        super(Net, self).__init__()
        self.no_repeat_ngram = P.NoRepeatNGram(ngram_size)

    def construct(self, s, l):
        return self.no_repeat_ngram(s, l)


state_seq = Tensor([[[1, 2, 1, 2, 5, 1, 2],
                     [9, 3, 9, 5, 4, 1, 5],
                     [4, 7, 9, 1, 9, 6, 1],
                     [7, 6, 4, 2, 9, 1, 5],
                     [7, 5, 8, 9, 9, 3, 9]],
                    [[7, 7, 2, 7, 9, 9, 4],
                     [3, 4, 7, 4, 7, 6, 8],
                     [1, 9, 5, 7, 6, 9, 3],
                     [4, 8, 6, 4, 5, 6, 4],
                     [4, 8, 8, 4, 3, 4, 8]]], dtype=mindspore.int32)

log_probs = Tensor(np.random.random((2, 5, 10)).astype(np.float32))
expect_log_probs = log_probs.asnumpy().copy()
expect_log_probs[0, 0, 1] = -FLT_MAX
expect_log_probs[0, 0, 5] = -FLT_MAX
expect_log_probs[1, 3, 5] = -FLT_MAX
expect_log_probs[1, 4, 8] = -FLT_MAX


def test_net():
    """
    Feature: test NoRepeatNGram on Ascend.
    Description: inputs with batch.
    Expectation: the result match with expect.
    """
    net = Net(ngram_size=3)
    output = net(state_seq, log_probs)

    print(expect_log_probs)
    print(output)
    assert np.array_equal(expect_log_probs, output.asnumpy())


def test_net_dynamic_shape():
    """
    Feature: test NoRepeatNGram dynamic shape on Ascend.
    Description: inputs with batch.
    Expectation: the result match with expect.
    """
    net = Net(ngram_size=3)
    place_holder_x = Tensor(shape=[None, 5, 7], dtype=mindspore.int32)
    place_holder_v = Tensor(shape=[None, 5, 10], dtype=mindspore.float32)
    net.set_inputs(place_holder_x, place_holder_v)
    output = net(state_seq, log_probs)
    assert np.array_equal(expect_log_probs, output.asnumpy())
