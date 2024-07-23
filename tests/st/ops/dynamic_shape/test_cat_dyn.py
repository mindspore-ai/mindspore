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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either matrix_inverseress or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
from tests.mark_utils import arg_mark

"""test cat dynamic shape"""

import pytest
import torch
import numpy as np
import mindspore as ms


class CatNet(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.cat_op = ms.ops.cat

    def construct(self, tensors, axis):
        out = self.cat_op(tensors, axis)
        return out


class CatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cat_op = torch.cat

    def forward(self, x, axis=0):
        out = self.cat_op(x, axis)
        return out


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cat_dynamic_shape():
    """
    Feature: test cat operator.
    Description: test dynamic cases for cat.
    Expectation: the result match with expect.
    """
    ms.set_context(mode=ms.GRAPH_MODE)
    input_x = ms.mutable([ms.Tensor(shape=(None, None), dtype=ms.float32) for _ in range(2)])
    axis = ms.mutable(input_data=1, dynamic_len=False)
    net = CatNet()
    net.set_inputs(input_x, axis)

    inputs2 = [ms.mutable([ms.Tensor(np.random.randn(3, 8), dtype=ms.float32) for _ in range(2)]), ms.mutable(0)]
    inputs3 = [ms.mutable([ms.Tensor(np.random.randn(12, 6), dtype=ms.float32) for _ in range(2)]), ms.mutable(-1)]
    for (x, axis) in [inputs2, inputs3]:
        out_ms = net(x, axis)
        tensors = [torch.from_numpy(i.asnumpy()) for i in x]
        out_torch = CatModule()(tensors, axis)
        assert np.allclose(out_ms.asnumpy(), out_torch.detach().numpy(), 0.01, 0.01)
