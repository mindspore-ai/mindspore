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

from mindspore import context
from mindspore import log as logger
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
from mindspore.nn import LayerNorm
from mindspore.ops.composite import GradOperation

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Grad(Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_, output_grad,):
        gout = self.grad(self.network)(input_, output_grad)
        return gout


class Net(Cell):
    def __init__(self, input_shape, begin_norm_axis, begin_params_axis, gamma, beta):
        super(Net, self).__init__()
        self.layernorm = LayerNorm(input_shape, begin_norm_axis, begin_params_axis, gamma, beta)

    def construct(self, input_):
        x = self.layernorm(input_)
        return x


def py_me_layernorm_grad(input_data, normalized_shape, gamma, beta, axis, gradients):
    input_me = Tensor(input_data)
    net_me = Grad(Net(normalized_shape, begin_norm_axis=axis,
                      begin_params_axis=axis,
                      gamma=Tensor(gamma),
                      beta=Tensor(beta)))
    net_me.set_train()
    out_pool_grad_me = Tensor(gradients)
    out_grad = net_me(input_me, out_pool_grad_me)
    logger.info("Check me result:")
    logger.info(out_grad.asnumpy())


def test_normal_layernorm_grad_normalize_2d():
    """
    1 input[1, 128, 1024],normalized_shape=[1024],element_affine=False
    """
    input_data = np.ones([1, 128, 1024]).astype(np.float32)
    gradients = np.ones((1, 128, 1024)).astype(np.float32)
    gamma = np.random.randn(1024).astype(np.float32)
    gamma.fill(1.1)
    beta = np.random.randn(1024).astype(np.float32)
    beta.fill(0.1)
    py_me_layernorm_grad(input_data, (1024,), gamma, beta, 2, gradients)
