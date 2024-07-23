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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

from mindspore import nn, Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_data_formata_dim_map():
    """
    Feature: Test the dynamic shape case of operator DataFormatDimMap.
    Description: Test the dynamic shape case of operator DataFormatDimMap and compare with expected output.
    Expectation: Output should be equal to expected value.
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.unique = P.Unique()
            self.reducesum = P.ReduceSum(keep_dims=False)
            self.data_format_dim_map = P.DataFormatDimMap()

        def construct(self, x, indices):
            x = self.data_format_dim_map(x.astype(mstype.int32))
            unique_indices, _ = self.unique(indices)
            x = self.reducesum(x.astype(mstype.int32), unique_indices)
            return self.data_format_dim_map(x.astype(mstype.int32))


    net = Net()
    x = [[1, 0, 0, -1, 0, -2, 1, 0],
         [0, 0, 1, -2, 0, 0, 1, -1],
         [0, 0, 0, 0, -1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, -1, 0, 1, 0],
         [0, 0, 0, 1, 0, 0, 0, 2],
         [0, 0, 0, 0, -1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0]]
    x = np.array(x)
    indices1 = np.random.randint(0, 1, (8,)).astype(np.int32)
    input_ms = Tensor(x.astype(np.int32))
    indices_ms = Tensor(indices1)
    input_dyn = Tensor(shape=[None for _ in input_ms.shape], dtype=input_ms.dtype)
    net.set_inputs(input_dyn, indices_ms)
    output = net(input_ms, indices_ms)
    output_expect = np.array([2, 0, 2, 1, 1, 2, 3, 2]).astype(np.int32)
    assert np.allclose(output.asnumpy(), output_expect)
