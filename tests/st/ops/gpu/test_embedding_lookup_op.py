# Copyright 2021 Huawei Technologies Co., Ltd
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

from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.nn as nn
import mindspore.context as context


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.embeddinglookup = P.EmbeddingLookup()

    def construct(self, input_params, input_indices, offset):
        return self.embeddinglookup(input_params, input_indices, offset)


def embeddinglookup_testcase(nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    input_params = Tensor(np.array([[8, 9], [10, 11], [12, 13], [14, 15]]).astype(nptype))
    input_indices = Tensor(np.array([[5, 2], [8, 5]]).astype(np.int32))
    offset = 4
    output = Net()(input_params, input_indices, offset)
    expect = np.array([[[10, 11], [0, 0]], [[0, 0], [10, 11]]]).astype(nptype)
    np.testing.assert_almost_equal(expect, output.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    input_params = Tensor(np.array([[8, 9], [10, 11], [12, 13], [14, 15]]).astype(nptype))
    input_indices = Tensor(np.array([[5, 2], [8, 5]]).astype(np.int32))
    offset = 4
    output = Net()(input_params, input_indices, offset)
    expect = np.array([[[10, 11], [0, 0]], [[0, 0], [10, 11]]]).astype(nptype)
    np.testing.assert_almost_equal(expect, output.asnumpy())


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_embeddinglookup_float32():
    embeddinglookup_testcase(np.float32)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_embeddinglookup_float16():
    embeddinglookup_testcase(np.float16)
