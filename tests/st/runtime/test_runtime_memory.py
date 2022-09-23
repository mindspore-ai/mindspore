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

import numpy as np
import pytest
from mindspore import context, nn, Tensor
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.common.parameter import Parameter
import mindspore.common.dtype as mstype


grad_all = C.GradOperation(get_all=True)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_while_grad_with_memory_optimize():
    """
    Feature: Integration of dynamic and static memory.
    Description: Test the control flow scene.
    Expectation: The result meet expectation.
    """
    class MyWhileNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.max = P.ReduceMax()

        def construct(self, idx, end, x):
            while idx < end:
                part = x[idx, :, :]
                max_num = self.max(part)
                x[idx, :, 0:2] = max_num
                idx = idx + 1
            return x

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    idx = Tensor(np.array(0), dtype=mstype.int32)
    end = Tensor(np.array(2), dtype=mstype.int32)
    input_x = np.array([[[4, 0], [0, 0]],
                        [[0, 4], [0, 0]]]).astype(np.float32)
    x = Tensor(input_x, dtype=mstype.float32)
    # memory optimize mode
    context.set_context(mode=context.GRAPH_MODE, memory_optimize_level="O1")
    while_net = MyWhileNet()
    net = GradNet(while_net)
    graph_output = net(idx, end, x)

    expect_zero = np.array([0], dtype=np.float32)
    expect_two = input_x
    assert np.allclose(graph_output[0].asnumpy(), expect_zero, 0.0001, 0.0001)
    assert np.allclose(graph_output[1].asnumpy(), expect_zero, 0.0001, 0.0001)
    assert np.allclose(graph_output[2].asnumpy(), expect_two, 0.0001, 0.0001)


class SparseApplyFtrlNet(nn.Cell):
    def __init__(self, var, accum, linear, lr=0.001, l1=0.0, l2=0.0, lr_power=-0.5):
        super(SparseApplyFtrlNet, self).__init__()
        self.sparse_apply_ftrl = P.SparseApplyFtrl(lr=lr, l1=l1, l2=l2, lr_power=lr_power)
        self.var = Parameter(var, name="var")
        self.accum = Parameter(accum, name="accum")
        self.linear = Parameter(linear, name="linear")

    def construct(self, grad, indices):
        out = self.sparse_apply_ftrl(self.var, self.accum, self.linear, grad, indices)
        return out


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sparse_apply_ftrl_with_memory_optimize():
    """
    Feature: Integration of dynamic and static memory.
    Description: Test the scene of output ref node.
    Expectation: The result meet expectation.
    """
    context.set_context(mode=context.GRAPH_MODE, memory_optimize_level="O1")

    grad_np = np.ones([3, 3, 3])
    indice_np = [0, 1, 2]
    var_np = np.ones([3, 3, 3])
    accum_np = np.ones([3, 3, 3])
    linear_np = np.ones([3, 3, 3])

    # test1: var/accum/linear/gradient are float32 and indices is int32.
    gradient = Tensor(grad_np, dtype=mstype.float32)
    indices = Tensor(indice_np, dtype=mstype.int32)
    var = Tensor(var_np, dtype=mstype.float32)
    accum = Tensor(accum_np, dtype=mstype.float32)
    linear = Tensor(linear_np, dtype=mstype.float32)
    sparse_apply_ftrl = SparseApplyFtrlNet(var, accum, linear)
    out = sparse_apply_ftrl(gradient, indices)
    expect_var = np.array([[[0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479]],
                           [[0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479]],
                           [[0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479],
                            [0.291479, 0.291479, 0.291479]]]).astype(np.float32)
    assert np.all(out[0].asnumpy() == expect_var)

    # test2: var/accum/linear/gradient are float16 and indices is int32.
    gradient = Tensor(grad_np, dtype=mstype.float16)
    indices = Tensor(indice_np, dtype=mstype.int32)
    var = Tensor(var_np, dtype=mstype.float16)
    accum = Tensor(accum_np, dtype=mstype.float16)
    linear = Tensor(linear_np, dtype=mstype.float16)
    sparse_apply_ftrl = SparseApplyFtrlNet(var, accum, linear)
    out = sparse_apply_ftrl(gradient, indices)
    expect_var = np.array([[[0.2915, 0.2915, 0.2915],
                            [0.2915, 0.2915, 0.2915],
                            [0.2915, 0.2915, 0.2915]],
                           [[0.2915, 0.2915, 0.2915],
                            [0.2915, 0.2915, 0.2915],
                            [0.2915, 0.2915, 0.2915]],
                           [[0.2915, 0.2915, 0.2915],
                            [0.2915, 0.2915, 0.2915],
                            [0.2915, 0.2915, 0.2915]]]).astype(np.float16)
    assert np.all(out[0].asnumpy() == expect_var)
