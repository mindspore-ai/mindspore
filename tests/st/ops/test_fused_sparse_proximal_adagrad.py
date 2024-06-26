# Copyright 2020 Huawei Technologies Co., Ltd
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
import mindspore.nn as nn
import mindspore.context as context
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.fused_sparse_proximal_adagrad = P.FusedSparseProximalAdagrad()
        self.var = Parameter(Tensor(np.ones([3, 3]).astype(np.float32)), name="var")
        self.accum = Parameter(Tensor(np.ones([3, 3]).astype(np.float32)), name="accum")
        self.lr = 0.01
        self.l1 = 0.0
        self.l2 = 0.0

    def construct(self, grad, indices):
        return self.fused_sparse_proximal_adagrad(self.var, self.accum, self.lr, self.l1, self.l2,
                                                  grad, indices)


def test_net():
    """
    Feature: FusedSparseProximalAdagrad
    Description: normal inputs
    Expectation: the result meets the expectation
    """
    gradient = Tensor(np.array([-3, 2, 3, 0, 0, 0, -4, -1, -2])
                      .reshape([3, 3]).astype(np.float32))
    indices = Tensor(np.ones([3]), mstype.int32)
    net = Net()
    output = net(gradient, indices)
    print(output)
    print(net.var.data)
    print(net.accum.data)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_fused_sparse_proximal_adagrad_dynamic():
    """
    Feature: FusedSparseProximalAdagrad
    Description: dynamic inputs
    Expectation: the result meets the expectation
    """

    class DynamicNet(nn.Cell):
        def __init__(self):
            super(DynamicNet, self).__init__()
            self.unique = P.Unique()
            self.gather = P.Gather()
            self.axis = 0

            self.sparse_apply_proximal_adagrad = P.FusedSparseProximalAdagrad()
            self.var = Parameter(Tensor(np.ones([3, 1, 2]).astype(np.float32)), name="var")
            self.accum = Parameter(Tensor(np.ones([3, 1, 2]).astype(np.float32)), name="accum")
            self.lr = Tensor(0.01, mstype.float32)
            self.l1 = Tensor(0.0, mstype.float32)
            self.l2 = Tensor(0.0, mstype.float32)

        def construct(self, grad, indices, indices_dy):
            indices_dy, _ = self.unique(indices_dy)
            grad = self.gather(grad, indices_dy, self.axis)
            indices = self.gather(indices, indices_dy, self.axis)
            out = self.sparse_apply_proximal_adagrad(self.var, self.accum, self.lr, self.l1,
                                                     self.l2, grad, indices)
            return out

    net = DynamicNet()
    grad = Tensor(np.array([[[0.1, 0.1]], [[0.1, 0.1]], [[0.1, 0.1]]]).astype(np.float32))
    indices = Tensor(np.array([0, 1, 2]).astype(np.int32))

    indices_dy = Tensor([0, 1], mstype.int32)
    net(grad, indices, indices_dy)
    print(net.var.data)
    expect_var = np.array([[[0.99900496, 0.99900496]],
                           [[0.99900496, 0.99900496]],
                           [[1., 1.]]]).astype(np.float32)
    assert np.allclose(net.var.data.asnumpy(), expect_var)
