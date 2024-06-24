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
import pytest

import mindspore.ops.operations as P
from mindspore import Tensor, nn, context
from mindspore.ops.composite import GradOperation
from mindspore.common.api import jit

context.set_context(device_target="Ascend")


class DynamicGRUV2(nn.Cell):

    def __init__(self):
        super(DynamicGRUV2, self).__init__()
        self.dynamic_gru = P.DynamicGRUV2()
        self.weight_i = Tensor(np.random.rand(64, 48).astype(np.float16))
        self.weight_h = Tensor(np.random.rand(16, 48).astype(np.float16))
        self.bias_i = Tensor(np.random.rand(48).astype(np.float16))
        self.bias_h = Tensor(np.random.rand(48).astype(np.float16))
        self.init_h = Tensor(np.random.rand(8, 16).astype(np.float16))

    def construct(self, x):
        return self.dynamic_gru(x, self.weight_i, self.weight_h, self.bias_i,
                                self.bias_h, None, self.init_h)[0]


class Grad(nn.Cell):

    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    @jit
    def construct(self, input_, output_grad):
        return self.grad(self.network)(input_, output_grad)


def dynamic_gru_v2_grad_test():
    x = Tensor(np.random.rand(2, 8, 64).astype(np.float16))
    sens = Tensor(np.random.rand(2, 8, 16).astype(np.float16))
    net = Grad(DynamicGRUV2())
    output = net(x, sens)
    print("***********x*********")
    print(x)

    print("***********output y*********")
    print(output[0].asnumpy())

    assert output[0].asnumpy().shape == x.asnumpy().shape


@pytest.mark.skip(reason="never run on ci or smoke test")
def test_dynamic_gru_v2_grad():
    """
    Feature: test DynamicGRUV2 Grad ops in ascend.
    Description: test the ops in static shape.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    dynamic_gru_v2_grad_test()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    dynamic_gru_v2_grad_test()
