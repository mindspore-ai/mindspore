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
import pytest

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.multifieldembeddinglookup = nn.MultiFieldEmbeddingLookup(10, 2, field_size=2, operator='SUM',
                                                                      target='DEVICE', dtype=ms.float16)

    def construct(self, x, y, z):
        out = self.multifieldembeddinglookup(x, y, z)
        return out


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_multifieldembeddinglookup_para_customed_dtype(mode):
    """
    Feature: MultiFieldEmbeddingLookup
    Description: Verify the result of MultiFieldEmbeddingLookup specifying customed para dtype.
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    input_indices = Tensor([[2, 4, 6, 0, 0], [1, 3, 5, 0, 0]], ms.int32)
    input_values = Tensor([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]], ms.float32)
    field_ids = Tensor([[0, 1, 1, 0, 0], [0, 0, 1, 0, 0]], ms.int32)
    output = net(input_indices, input_values, field_ids)
    expect_output_shape = (2, 2, 2)
    assert np.allclose(expect_output_shape, output.shape)
