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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import numpy as np
from tests.mark_utils import arg_mark
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, Parameter, ops


class Net(nn.Cell):
    def __init__(self, var1, m1, v1, var2, m2, v2, var3, m3, v3, var4,
                 m4, v4, var5, m5, v5, var6, m6, v6, var7, m7, v7):
        super(Net, self).__init__()
        self.adam_weight_decay = ops.AdamWeightDecay(use_locking=False)
        self.var1 = Parameter(Tensor(var1), name="var1")
        self.m1 = Parameter(Tensor(m1), name="m1")
        self.v1 = Parameter(Tensor(v1), name="v1")
        self.var2 = Parameter(Tensor(var2), name="var2")
        self.m2 = Parameter(Tensor(m2), name="m2")
        self.v2 = Parameter(Tensor(v2), name="v2")
        self.var3 = Parameter(Tensor(var3), name="var3")
        self.m3 = Parameter(Tensor(m3), name="m3")
        self.v3 = Parameter(Tensor(v3), name="v3")
        self.var4 = Parameter(Tensor(var4), name="var4")
        self.m4 = Parameter(Tensor(m4), name="m4")
        self.v4 = Parameter(Tensor(v4), name="v4")
        self.var5 = Parameter(Tensor(var5), name="var5")
        self.m5 = Parameter(Tensor(m5), name="m5")
        self.v5 = Parameter(Tensor(v5), name="v5")
        self.var6 = Parameter(Tensor(var6), name="var6")
        self.m6 = Parameter(Tensor(m6), name="m6")
        self.v6 = Parameter(Tensor(v6), name="v6")
        self.var7 = Parameter(Tensor(var7), name="var7")
        self.m7 = Parameter(Tensor(m7), name="m7")
        self.v7 = Parameter(Tensor(v7), name="v7")

    def construct(self, lr, beta1, beta2, epsilon, decay, grad):
        out1 = self.adam_weight_decay(self.var1, self.m1, self.v1, lr, beta1, beta2,
                                      epsilon, decay, grad)
        out2 = self.adam_weight_decay(self.var2, self.m2, self.v2, lr, beta1, beta2,
                                      epsilon, decay, grad)
        out3 = self.adam_weight_decay(self.var3, self.m3, self.v3, lr, beta1, beta2,
                                      epsilon, decay, grad)
        out4 = self.adam_weight_decay(self.var4, self.m4, self.v4, lr, beta1, beta2,
                                      epsilon, decay, grad)
        out5 = self.adam_weight_decay(self.var5, self.m5, self.v5, lr, beta1, beta2,
                                      epsilon, decay, grad)
        out6 = self.adam_weight_decay(self.var6, self.m6, self.v6, lr, beta1, beta2,
                                      epsilon, decay, grad)
        out7 = self.adam_weight_decay(self.var7, self.m7, self.v7, lr, beta1, beta2,
                                      epsilon, decay, grad)
        return out1, out2, out3, out4, out5, out6, out7


def get_output(gradient, var1, v1, m1, var2, v2, m2, var3, v3, m3, var4,
               v4, m4, var5, v5, m5, var6, v6, m6, var7, v7, m7, enable_graph_kernel=False):
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=enable_graph_kernel)
    if enable_graph_kernel:
        context.set_context(enable_graph_kernel=True,
                            graph_kernel_flags="--enable_parallel_fusion=True "
                                               "--enable_expand_ops=AdamApplyOneWithDecay")
    net = Net(var1, v1, m1, var2, v2, m2, var3, v3, m3, var4,
              v4, m4, var5, v5, m5, var6, v6, m6, var7, v7, m7)
    output = net(Tensor(0.001), Tensor(0.9), Tensor(0.999), Tensor(1e-8), Tensor(0.0), gradient)
    return output


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_parallel_fusion_ascend():
    """
    Feature: test graph kernel parallel
    Description: run test case on Ascend
    Expectation: the result matches the expected result
    """
    context.set_context(jit_level='O0')
    var1 = np.random.random([2, 2]).astype(np.float32)
    var2 = np.random.random([2, 2]).astype(np.float32)
    var3 = np.random.random([2, 2]).astype(np.float32)
    var4 = np.random.random([2, 2]).astype(np.float32)
    var5 = np.random.random([2, 2]).astype(np.float32)
    var6 = np.random.random([2, 2]).astype(np.float32)
    var7 = np.random.random([2, 2]).astype(np.float32)
    v1 = np.random.random([2, 2]).astype(np.float32)
    v2 = np.random.random([2, 2]).astype(np.float32)
    v3 = np.random.random([2, 2]).astype(np.float32)
    v4 = np.random.random([2, 2]).astype(np.float32)
    v5 = np.random.random([2, 2]).astype(np.float32)
    v6 = np.random.random([2, 2]).astype(np.float32)
    v7 = np.random.random([2, 2]).astype(np.float32)
    m1 = np.random.random([2, 2]).astype(np.float32)
    m2 = np.random.random([2, 2]).astype(np.float32)
    m3 = np.random.random([2, 2]).astype(np.float32)
    m4 = np.random.random([2, 2]).astype(np.float32)
    m5 = np.random.random([2, 2]).astype(np.float32)
    m6 = np.random.random([2, 2]).astype(np.float32)
    m7 = np.random.random([2, 2]).astype(np.float32)
    input_array = np.random.random([2, 2]).astype(np.float32)
    expect = get_output(Tensor(input_array), var1, v1, m1, var2, v2, m2, var3,
                        v3, m3, var4, v4, m4, var5, v5, m5, var6, v6, m6, var7, v7, m7, False)
    output = get_output(Tensor(input_array), var1, v1, m1, var2, v2, m2, var3,
                        v3, m3, var4, v4, m4, var5, v5, m5, var6, v6, m6, var7, v7, m7, True)
    for i in range(7):
        for j in range(3):
            expect_np = expect[i][j].asnumpy().copy()
            output_np = output[i][j].asnumpy().copy()
            assert np.allclose(expect_np, output_np)
