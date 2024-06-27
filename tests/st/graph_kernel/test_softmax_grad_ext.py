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
from tests.mark_utils import arg_mark
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.Mul = P.Mul()
        self.ReduceSum = P.ReduceSum(keep_dims=True)
        self.Sub = P.Sub()
        self.axes = (0)

    def construct(self, x, y, z):
        mul = self.Mul(y, x)
        reduce_sum = self.ReduceSum(mul, self.axes)
        sub = self.Sub(x, reduce_sum)
        mul1 = self.Mul(y, sub)
        mul_grad = self.Mul(mul1, z)
        return mul_grad


def get_output(x, y, z, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net = Net()
    output = net(x, y, z)
    return output


def softmax_grad_ext_test(shape1, dtype):
    x = Tensor(np.random.normal(0, 10, shape1).astype(dtype))
    y = Tensor(np.random.normal(0, 10, shape1).astype(dtype))
    z = Tensor(np.random.normal(0, 10, shape1).astype(dtype))
    expect = get_output(x, y, z, False)
    output = get_output(x, y, z, True)

    expect_np = expect.asnumpy().copy()
    output_np = output.asnumpy().copy()

    assert np.allclose(expect_np, output_np, 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ascend():
    """
    Feature: Graph Kernel expander
    Description: Verify SoftmaxGradExt expander in Ascend
    Expectation: No exception
    """
    context.set_context(mode=context.GRAPH_MODE, graph_kernel_flags="--enable_expand_ops=SoftmaxGradExt")
    softmax_grad_ext_test((4, 4), np.float32)
