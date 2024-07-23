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
import mindspore.context as context
from mindspore import nn
from mindspore import Tensor
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops import composite as C
from mindspore.ops import operations as P

context.set_context(device_target="Ascend")


class NetGrad(nn.Cell):
    def __init__(self):
        super(NetGrad, self).__init__()
        self.grad = G.StridedSliceGrad()

    def construct(self, x, begin, end, strides, dout):
        return self.grad(x, begin, end, strides, dout)


class NetGradGrad(nn.Cell):
    def __init__(self, forward_net):
        super(NetGradGrad, self).__init__()
        self.forward_net = forward_net
        self.grad_ops = C.GradOperation(get_all=True, sens_param=True)

    def construct(self, dy, shapex, begin, end, strides, dout):
        backward_net = self.grad_ops(self.forward_net)
        return backward_net(dy, shapex, begin, end, strides, dout)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_stridedslice_high_grad_float32():
    """
    Feature: StridedSlice Grad Grad operation
    Description: test the grad of StridedSliceGrad kernel, with float input.
    Expectation: the output is same with numpy
    """
    x = np.array([[[1, 1, 1], [2, 2, 2]],
                  [[3, 3, 3], [4, 4, 4]],
                  [[5, 5, 5], [6, 6, 6]]]).astype(np.float32)

    dy = Tensor(np.ones((2, 1, 1)).astype(np.float32))
    x_shape = Tensor(np.array(list(x.shape)).astype(np.int64))
    begin = (1, 0, 2)
    end = (3, 1, 3)
    strides = (1, 1, 1)
    dout = np.ones_like(x).astype(np.float32)

    grad_net = NetGrad()
    grad_grad_net = NetGradGrad(grad_net)
    dgrad_ms = grad_grad_net(dy, x_shape, begin, end, strides, Tensor(dout))

    stridedslice = P.StridedSlice()
    forward_res = stridedslice(Tensor(x), begin, end, strides)
    expected = np.ones_like(forward_res.asnumpy())
    assert np.allclose(dgrad_ms[0].asnumpy(), expected, 1e-4, 1e-4)
