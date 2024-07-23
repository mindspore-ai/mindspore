# Copyright 2024 Huawei Technologies Co., Ltd
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
from mindspore import ops
from mindspore import nn
from mindspore import Tensor
from mindspore import dtype as mstype
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    def __init__(self, nbins):
        super(Net, self).__init__()
        self.op = ops.HistogramFixedWidth(nbins=nbins)

    def construct(self, x, range_op):
        return self.op(x, range_op)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_histogram_fixed_width_float64():
    """
    Feature: HistogramFixedWidth op
    Description: The input type of the HistogramFixedWidth operator is float64
    Expectation: The result match to the expect value
    """
    nbins = 5
    net = Net(nbins)
    x = Tensor([-1.0, 0.0, 1.5, 2.0, 5.0, 15], mstype.float64)
    range_op = Tensor([0.0, 5.0], mstype.float64)
    output = net(x, range_op)
    expected_output = np.array([2, 1, 1, 0, 2])
    assert np.array_equal(output.asnumpy(), expected_output)
