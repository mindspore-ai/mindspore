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


class NetApproximateEqual(nn.Cell):
    def __init__(self, tolerance):
        super(NetApproximateEqual, self).__init__()
        self.tolerance = tolerance
        self.approximate_equal = P.ApproximateEqual(self.tolerance)

    def construct(self, input_x1, input_x2):
        output = self.approximate_equal(input_x1, input_x2)
        return output


@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_approximate_equal_fp16():
    """
    Feature: ALL To ALL
    Description: test cases for ApproximateEqual
    Expectation: the result match to expect result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    approximate_equal = NetApproximateEqual(tolerance=0.01)
    x1 = Tensor([1., 2., 3., 4.], mindspore.float16)
    x2 = Tensor([1., 2., 3., 4.], mindspore.float16)
    output = approximate_equal(x1, x2)
    expect = Tensor(([True, True, True, True]), mindspore.bool_)
    assert (output.asnumpy() == expect).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_approximate_equal_fp32():
    """
    Feature: ALL To ALL
    Description: test cases for ApproximateEqual
    Expectation: the result match to expect result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    approximate_equal = NetApproximateEqual(tolerance=0.07)
    x1 = Tensor([1., 2., 3., 4.], mindspore.float32)
    x2 = Tensor([3., 1., 3., 1.], mindspore.float32)
    output = approximate_equal(x1, x2)
    expect = Tensor(([False, False, True, False]), mindspore.bool_)
    assert (output.asnumpy() == expect).all()
