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

import pytest
import torch
import numpy as np
import mindspore as ms
import mindspore.nn as nn

from mindspore import Tensor, ops, context


class AffineGridNet(nn.Cell):
    def __init__(self, align_corners=False):
        super(AffineGridNet, self).__init__()
        self.affine_grid = ops.AffineGrid(align_corners=align_corners)

    def construct(self, theta, size):
        return self.affine_grid(theta, size)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_affine_grid_corner_case():
    """
    Feature: gpu backend of operator AffineGrid
    Description: special case when h or w = 1 and align_corners = True
    Expectation: success
    """
    n, c, h, w = 2, 2, 2, 1
    net = AffineGridNet(align_corners=True)
    t = Tensor(np.ones((2, 2, 3)), ms.float32)
    output_size = (n, c, h, w)
    output = net(t, output_size)
    expected = np.array([[[[0, 0]], [[2, 2]]],
                         [[[0, 0]], [[2, 2]]]]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expected, atol=0.001, rtol=0.001)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('align', [False, True])
@pytest.mark.parametrize('dtype', [ms.float32, ms.float16])
def test_affine_grid_4d_normal(mode, align, dtype):
    """
    Feature: gpu backend of operator AffineGrid
    Description: normal case AffineGrid4D
    Expectation: success
    """
    n, c, h, w = 2, 2, 2, 3
    theta = np.ones((n, 2, 3)).astype(np.float32)
    out_size = (n, c, h, w)
    ms_input = Tensor(theta, dtype)
    torch_input = torch.tensor(theta, dtype=torch.float32)
    net = AffineGridNet(align_corners=align)
    output = net(ms_input, out_size)
    expected_output = torch.nn.functional.affine_grid(torch_input, list(out_size), align_corners=align)
    assert np.allclose(output.asnumpy(), expected_output, atol=0.001, rtol=0.001)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('align', [False, True])
@pytest.mark.parametrize('dtype', [ms.float32, ms.float16])
def test_affine_grid_5d_normal(mode, align, dtype):
    """
    Feature: gpu backend of operator AffineGrid
    Description: normal case AffineGrid5D
    Expectation: success
    """
    n, c, d, h, w = 2, 3, 4, 2, 3
    theta = np.ones((n, 3, 4)).astype(np.float32)
    out_size = (n, c, d, h, w)
    ms_input = Tensor(theta, dtype)
    torch_input = torch.tensor(theta, dtype=torch.float32)
    net = AffineGridNet(align_corners=align)
    output = net(ms_input, out_size)
    expected_output = torch.nn.functional.affine_grid(torch_input, list(out_size), align_corners=align)
    assert np.allclose(output.asnumpy(), expected_output, atol=0.001, rtol=0.001)
