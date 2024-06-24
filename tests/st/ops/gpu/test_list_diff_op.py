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

import numpy as np
import pytest

from mindspore.common import dtype as mstype
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations.array_ops import ListDiff


class NetListDiff(nn.Cell):
    def __init__(self, out_idx=mstype.int64):
        super(NetListDiff, self).__init__()
        self.list_diff = ListDiff(out_idx=out_idx)

    def construct(self, x, y):
        return self.list_diff(x, y)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_diff_int32():
    """
    Feature: ListDiff gpu TEST.
    Description: 1d test case for ListDiff
    Expectation: the result match to expect
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.array([1, 2, 3, 4, 5, 6]).astype(np.int32)
    y = np.array([1, 3, 6]).astype(np.int32)
    res_out = np.array([2, 4, 5]).astype(np.int32)
    res_idx = np.array([1, 3, 4]).astype(np.int64)
    x1 = Tensor(x)
    y1 = Tensor(y)
    net = NetListDiff(out_idx=mstype.int64)
    out, idx = net(x1, y1)
    assert np.allclose(res_out, out.asnumpy())
    assert np.allclose(res_idx, idx.asnumpy())


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_diff_fp32():
    """
    Feature: ListDiff gpu TEST.
    Description: 1d test case for ListDiff
    Expectation: the result match to expect
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.array([1.5, 2.0, 3.1, 4.5, 5, 6]).astype(np.float32)
    y = np.array([1.5, 3.1, 6]).astype(np.float32)
    res_out = np.array([2.0, 4.5, 5]).astype(np.float32)
    res_idx = np.array([1, 3, 4]).astype(np.int64)
    x1 = Tensor(x)
    y1 = Tensor(y)
    net = NetListDiff(out_idx=mstype.int64)
    out, idx = net(x1, y1)
    assert np.allclose(res_out, out.asnumpy())
    assert np.allclose(res_idx, idx.asnumpy())
