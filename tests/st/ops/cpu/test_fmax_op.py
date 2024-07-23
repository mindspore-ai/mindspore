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

"""test Fmax forward and backward broadcast"""

import pytest
import numpy as np

import mindspore
from mindspore import context
from mindspore import Tensor, ops
from mindspore import nn
from mindspore.ops import composite as C

class FmaxForward(nn.Cell):
    def __init__(self):
        super().__init__()
        self.fmax = ops.fmax

    def construct(self, x1, x2):
        return self.fmax(x1, x2)


class FmaxGrad(nn.Cell):
    def __init__(self, forward):
        super().__init__()
        self.forward = forward
        self.grad = C.GradOperation(get_all=True)

    def construct(self, x1, x2):
        return self.grad(self.forward)(x1, x2)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('dtype', [np.int, np.float32, np.float64])
@pytest.mark.parametrize("context_mode", [mindspore.GRAPH_MODE, mindspore.PYNATIVE_MODE])
def test_fmax_cpu_broadcast_shape(dtype, context_mode):
    """
    Feature: test fmax op forward and backward.
    Description: test the ops in broadcast mode.
    Expectation: expect correct output shape.
    """
    context.set_context(mode=context_mode, device_target="CPU")
    fmax = FmaxForward()
    x1 = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).astype(dtype))
    x2 = Tensor(np.array([4]).astype(dtype))
    fmax.set_inputs(x1, x2)
    y = fmax(x1, x2)
    expect = np.array([[4, 4, 4, 4], [5, 6, 7, 8]]).astype(dtype)
    assert (y.asnumpy() == expect).all()
    fmax_grad = FmaxGrad(FmaxForward())
    out_grad = fmax_grad(x1, x2)
    expect_grad = np.array([[0, 0, 0, 1],
                            [1, 1, 1, 1]]).astype(dtype)
    assert np.allclose(out_grad[0].asnumpy(), expect_grad)
