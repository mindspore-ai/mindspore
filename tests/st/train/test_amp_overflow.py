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

'''test overflow'''
import pytest
import numpy as np

from mindspore import Tensor, Parameter, nn, ops
import mindspore.amp as amp
import mindspore as ms


class Net(nn.Cell):
    def __init__(self, in_features, out_features):
        super(Net, self).__init__()
        self.weight = Parameter(Tensor(np.full([in_features, out_features], 2, np.float16)),
                                name='weight')
        self.matmul = ops.MatMul()

    def construct(self, x):
        output = self.matmul(x, self.weight)
        return output


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_functional_amp_overflow(mode):
    """
    Feature: mindspore.amp.overflow
    Description: test amp overflow
    Expectation: Success.
    """
    ms.set_context(mode=mode)
    size, in_features, out_features = 1, 2, 2
    net = Net(in_features, out_features)
    loss_fn = nn.MSELoss()

    def forward_fn(data, label):
        logits = net(data)
        loss = loss_fn(logits, label)
        return loss, logits

    grad_fn = ops.value_and_grad(forward_fn, grad_position=None, weights=net.trainable_params())

    @ms.jit
    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        is_finite = amp.all_finite(grads)
        return loss, is_finite

    shape = (size, in_features)
    inputs = [
        Tensor(np.full(shape, -np.inf, np.float16)),
        Tensor(np.full(shape, 0, np.float16)),
        Tensor(np.full(shape, 40000, np.float16)),
        Tensor(np.full(shape, 10, np.float16)),
        Tensor(np.full(shape, np.inf, np.float16)),
    ]
    label = Tensor(np.full([out_features,], 0, np.float16))
    datasets = list(zip(inputs, [label for _ in range(len(inputs))]))
    expect_results = [False, True, False, True, False]
    outputs = []
    for data, label in datasets:
        _, is_finite = train_step(data, label)
        outputs.append(is_finite.asnumpy().tolist())
    assert outputs == expect_results
