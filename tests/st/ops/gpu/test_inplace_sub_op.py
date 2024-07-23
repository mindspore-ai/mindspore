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
from tests.mark_utils import arg_mark

# This example should be run with multiple processes.

# Please refer to Learning > Tutorials > Experts > Distributed Parallel Startup Methods on mindspore.cn. Pick a

# supported startup method for your hardware and get more detail information on the corresponding page.

import pytest

import mindspore
from mindspore import context
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore import nn

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class NetInplaceSub(nn.Cell):
    def __init__(self, indices):
        super(NetInplaceSub, self).__init__()
        self.indices = indices
        self.inplace_sub = P.InplaceSub(self.indices)

    def construct(self, input_x1, input_x2):
        output = self.inplace_sub(input_x1, input_x2)
        return output


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_inplace_sub_fp16():
    """
    Feature: ALL To ALL
    Description: test cases for InplaceSub
    Expectation: the result match to expect result
    """
    inplace_sub = NetInplaceSub(indices=(0, 1))
    x1 = Tensor([[1, 2], [3, 4], [5, 6]], mindspore.float16)
    x2 = Tensor([[0.5, 1.0], [1.0, 1.5]], mindspore.float16)
    output = inplace_sub(x1, x2)
    expect = Tensor([[0.5, 1.0], [2.0, 2.5], [5.0, 6.0]], mindspore.float16)
    assert (output.asnumpy() == expect).all()
