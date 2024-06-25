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
"""test cell tensor in list with PIJit and pynative mode"""

from mindspore import nn, Tensor
from mindspore import dtype as mstype
from mindspore import jit, context
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.list = [Tensor([1], mstype.int32), Tensor([2], mstype.int32), Tensor([3], mstype.int32)]

    def construct(self, c):
        if c in self.list:
            out = c + c
        else:
            out = c + 0
        return out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_in_list():
    """
    Feature:  tensor list.
    Description: test tensor in list for parameter in construct.
    Expectation: No exception.
    """
    net = Net()
    context.set_context(mode=context.PYNATIVE_MODE)
    jit(net.construct, mode="PIJit")

    output = net(Tensor([1], mstype.int32))
    expect = Tensor([2], mstype.int32)
    assert output == expect

    output = net(Tensor([2], mstype.int32))
    expect = Tensor([4], mstype.int32)
    assert output == expect

    output = net(Tensor([3], mstype.int32))
    expect = Tensor([6], mstype.int32)
    assert output == expect

    output = net(Tensor([4], mstype.int32))
    expect = Tensor([4], mstype.int32)
    assert output == expect
