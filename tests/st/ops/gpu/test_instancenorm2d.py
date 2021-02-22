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
from mindspore.common.api import ms_function
from mindspore.ops import functional as F
from mindspore.ops.composite import GradOperation
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    @ms_function
    def construct(self, input_x, grad):
        return self.grad(self.network)(input_x, grad)

class Net(nn.Cell):
    def __init__(self, n):
        super(Net, self).__init__()
        self.ops = nn.BatchNorm2d(n, use_batch_statistics=True, gamma_init=0.5, beta_init=0.5)

    def construct(self, x):
        shape = F.shape(x)
        return F.reshape(self.ops(F.reshape(x, (1, -1, shape[2], shape[3]))), shape)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_InstanceNorm2d_fp32():
    x_np = np.random.randn(3, 3, 2, 2).astype(np.float32)
    bn_instance_comp = Net(3 * 3)
    bn_instance_op = nn.InstanceNorm2d(3, use_batch_statistics=True, gamma_init=0.5, beta_init=0.5)
    comp_out = bn_instance_comp(Tensor(x_np))
    op_out = bn_instance_op(Tensor(x_np))
    assert np.allclose(comp_out.asnumpy(), op_out.asnumpy())

    sens = np.random.randn(3, 3, 2, 2).astype(np.float32)
    bn_comp_backward_net = Grad(bn_instance_comp)
    bn_op_backward_net = Grad(bn_instance_op)
    output1 = bn_comp_backward_net(Tensor(x_np), Tensor(sens))
    output2 = bn_op_backward_net(Tensor(x_np), Tensor(sens))
    assert np.allclose(output1[0].asnumpy(), output2[0].asnumpy())
