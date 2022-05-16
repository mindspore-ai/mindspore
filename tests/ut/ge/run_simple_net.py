# Copyright 2022 Huawei Technologies Co., Ltd
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
from mindspore import Tensor
from mindspore import context
from mindspore.ops import operations as P
from mindspore.nn import Cell

context.set_context(mode=context.GRAPH_MODE)


class SeqNet(Cell):
    def __init__(self):
        super().__init__()
        self.op_seq = (P.Sqrt(), P.Reciprocal(), P.Square())

    def construct(self, x):
        t = x
        for op in self.op_seq:
            t = op(t)
        return t


def test_op_seq_net_ge():
    """
    Feature: unify ge and vm backend
    Description: test op seq with ge backend
    Expectation: success
    """
    net = SeqNet()
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    net(input_me)


class TriuNet(nn.Cell):
    def __init__(self):
        super(TriuNet, self).__init__()
        self.value = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def construct(self):
        triu = nn.Triu()
        return triu(self.value, 0)


def test_triu_ge():
    """
    Feature: unify ge and vm backend
    Description: test TriuNet with ge backend
    Expectation: success
    """
    net = TriuNet()
    out = net()
    assert np.sum(out.asnumpy()) == 26

if __name__ == "__main__":
    test_op_seq_net_ge()
    test_triu_ge()
