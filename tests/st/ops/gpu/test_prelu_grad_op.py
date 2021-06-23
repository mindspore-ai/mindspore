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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import _grad_ops as G


class NetPReLUGrad(nn.Cell):
    def __init__(self):
        super(NetPReLUGrad, self).__init__()
        self.prelu_grad = G.PReLUGrad()

    def construct(self, dout, x, w):
        return self.prelu_grad(dout, x, w)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_prelu_grad_fp32_channel_shared():
    dout = Tensor(np.ones(shape=[2, 2, 2, 3]).astype(np.float32))
    x = Tensor(np.arange(-5, 19).reshape(2, 2, 2, 3).astype(np.float32))
    w = Tensor(np.array([-0.5]).astype(np.float32))
    expect_dx = np.array([[[[-0.5000, -0.5000, -0.5000],
                            [-0.5000, -0.5000, -0.5000]],
                           [[1.0000, 1.0000, 1.0000],
                            [1.0000, 1.0000, 1.0000]]],
                          [[[1.0000, 1.0000, 1.0000],
                            [1.0000, 1.0000, 1.0000]],
                           [[1.0000, 1.0000, 1.0000],
                            [1.0000, 1.0000, 1.0000]]]]).astype(np.float32)
    expect_dw = np.array([-15.]).astype(np.float32)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    prelu_grad = NetPReLUGrad()
    dx, dw = prelu_grad(dout, x, w)
    assert (dx.asnumpy() == expect_dx).all()
    assert (dw.asnumpy() == expect_dw).all()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    prelu_grad = NetPReLUGrad()
    dx, dw = prelu_grad(dout, x, w)
    assert (dx.asnumpy() == expect_dx).all()
    assert (dw.asnumpy() == expect_dw).all()
