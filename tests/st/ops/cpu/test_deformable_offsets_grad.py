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
import numpy as np

from mindspore.ops.operations import _grad_ops as G
from mindspore.ops import composite as C
from mindspore import nn
from mindspore import context
from mindspore import Tensor
from mindspore import dtype
from mindspore.ops.operations import nn_ops

grad_all = C.GradOperation(get_all=True)


class TestNetwork(nn.Cell):
    def __init__(self):
        super(TestNetwork, self).__init__()
        stride = (1, 1, 1, 1)
        pad = (0, 0, 0, 0)
        ksize = (2, 2)
        self.deformable_offsets_grad_op = G.DeformableOffsetsGrad(stride, pad, ksize)

    def construct(self, dout, x, offsets):
        output = self.deformable_offsets_grad_op(dout, x, offsets)
        return output


def test_grad_infer():
    """
    Feature: CPU operation.
    Description: Test of CPU operation: DeformableOffsetsGrad
    Expectation: No exception raised.
    """
    context.set_context(save_graphs=True, save_graphs_path="./graph_ir")
    dout = Tensor(np.ones([1, 1, 2, 2]), dtype.float32)
    x = Tensor(np.ones([1, 1, 2, 2]), dtype.float32)
    offsets = Tensor(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                               0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                               0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                               0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]).astype(np.float32)
                     .reshape([1, 12, 2, 2]))
    net = TestNetwork()
    grad = net(dout, x, offsets)
    print("grad_x:", grad[0])
    print("grad_offset:", grad[1])
    return grad


class ForwardNet(nn.Cell):
    def __init__(self):
        super(ForwardNet, self).__init__()
        stride = (1, 1, 1, 1)
        pad = (0, 0, 0, 0)
        ksize = (2, 2)
        self.deformable_offsets_grad_op = nn_ops.DeformableOffsets(stride, pad, ksize)

    def construct(self, x, offsets):
        output = self.deformable_offsets_grad_op(x, offsets)
        return output


class BackwardNet(nn.Cell):
    def __init__(self, net):
        super(BackwardNet, self).__init__()
        self.net = net

    def construct(self, *inputs):
        out = self.net(*inputs)
        return out, grad_all(self.net)(*inputs)


def test_auto_diff():
    """
    Feature: CPU operation.
    Description: Test of CPU operation: DeformableOffsetsGrad by auto diff.
    Expectation: No exception raised.
    """
    context.set_context(save_graphs=True, save_graphs_path="./graph_ir")
    x = Tensor(np.ones([1, 1, 2, 2]), dtype.float32)
    offsets = Tensor(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                               0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                               0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                               0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]).astype(np.float32)
                     .reshape([1, 12, 2, 2]))
    forward_net = ForwardNet()
    net = BackwardNet(forward_net)
    grad = net(x, offsets)
    print("grad_x:", grad[0])
    print("grad_offset:", grad[1])
    return grad
