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
""" test_tensor_in_list """
from tests.st.compiler.control.cases_register import case_register
from mindspore import nn, Tensor
from mindspore import dtype as mstype


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


@case_register.level1
@case_register.target_ascend
def test_tensor_in_list():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    net = Net()
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
