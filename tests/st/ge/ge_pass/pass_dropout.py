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
""" test ge frontend pass `DropoutForGE` `DropoutGradForGE` """
import numpy as np

from tests.st.ge import ge_infer_env  # pylint: disable=unused-import
from mindspore import ops, nn, context, Tensor
from mindspore.ops.composite import GradOperation


class DropoutNet(nn.Cell):
    def __init__(self, keep_prob):
        super(DropoutNet, self).__init__()
        self.drop = nn.Dropout(p=1.0 - keep_prob)
        self.relu = ops.ReLU()

    def construct(self, x):
        x = self.relu(x)
        return self.relu(self.drop(x))


class _Grad(nn.Cell):
    def __init__(self, grad, network):
        super().__init__()
        self.network = network
        self.grad = grad

    def construct(self, *inputs):
        return self.grad(self.network)(*inputs)


class GradOfFirstInput(_Grad):
    """
    get grad of first input
    """

    def __init__(self, network, sens_param=True):
        super().__init__(grad=GradOperation(sens_param=sens_param), network=network)


def ge_drop_out_0_5(shape):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    net = DropoutNet(0.5)
    net.set_train()
    x = Tensor(np.ones(shape).astype(np.float32))
    out = net(x)
    return out


def ge_dropout_backward_0_5(shape):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    net = DropoutNet(0.5)
    grad_net = GradOfFirstInput(net)
    grad_net.set_train()
    x = Tensor(np.ones(shape).astype(np.float32))
    sens = Tensor(np.ones(shape).astype(np.float32))
    out = grad_net(x, sens)
    return out


def run_ge_dropout():
    """
    Feature: Test Dropout in GE backend.
    Description: Test Dropout in GE backend.
    Expectation: Dropout result is random, assert result shape.
    """
    shape = (1, 1, 6, 6)
    out = ge_drop_out_0_5(shape)
    assert out.asnumpy().shape == shape


def run_ge_dropout_backward():
    """
    Feature: Test Dropout backward in GE backend.
    Description: Test Dropout backward in GE backend.
    Expectation: Dropout result is random, assert gradient shape.
    """
    shape = (1, 1, 6, 6)
    out = ge_dropout_backward_0_5(shape)
    assert out.asnumpy().shape == shape


if __name__ == "__main__":
    run_ge_dropout()
    run_ge_dropout_backward()
